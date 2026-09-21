"""Hybrid simulation implementation."""

from __future__ import annotations

from collections.abc import Mapping
from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from hybrid_ao_psf import (
    DiagnosticsLevel,
    HybridRequest,
    HybridResult,
    NgsHoMetricInterpolator,
    NgsMetricProvider,
    NonPsdPolicy,
    ScienceHoPsfInterpolator,
    SciencePsfProvider,
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
    simulate,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
    validate_science_ho_psf_interpolator,
    validate_science_ho_psf_query,
)

from .._units import quantity_value, unit_string
from . import atm, schema
from .base import BaseSimulationSetup, PsfParameters
from .coordinates import polar_to_cartesian
from .interfaces import SimulationContext, SimulationSetup
from .photometry import magnitudes_to_photons_per_frame
from .tiptop_config_backed import (
    TiptopConfigBackedSimulation,
    _serialize_parser,
)
from .validation import normalize_meta_fields


@dataclass(frozen=True)
class HybridSetup(BaseSimulationSetup):
    """Typed setup payload for ``HybridSimulation``.

    This setup currently adds no fields beyond ``BaseSimulationSetup``. It
    gives the upstream Hybrid runtime its own type without inheriting
    from TIPTOP runtime setup.
    """


@dataclass(frozen=True)
class HybridDiagnosticsContext:
    """Upstream result and AO slot mapping available to diagnostic hooks.

    Attributes:
        result: Immutable result returned by Hybrid AO PSF.
        ngs_used: Slot-aligned mask of active NGS stars for this option row.
    """

    result: HybridResult
    ngs_used: np.ndarray


@dataclass(frozen=True)
class HybridResolvedInputs:
    """Atomic upstream inputs returned by the downstream extension seam.

    Subclasses may call ``HybridSimulation._resolve_hybrid_inputs()`` and
    replace instrument-owned request values or wrap a provider. The upstream
    engine call and AO result mapping remain owned by ``HybridSimulation``.

    Attributes:
        request: Fully resolved request for one upstream simulation.
        science_provider: Loaded reusable science-HO-PSF provider.
        ngs_provider: Loaded reusable NGS-HO-metric provider.
        ngs_used: AO slot mask corresponding to the active request vectors.
    """

    request: HybridRequest
    science_provider: SciencePsfProvider
    ngs_provider: NgsMetricProvider
    ngs_used: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.request, HybridRequest):
            raise TypeError("request must be a HybridRequest.")
        if not isinstance(self.science_provider, SciencePsfProvider):
            raise TypeError("science_provider must implement SciencePsfProvider.")
        if not isinstance(self.ngs_provider, NgsMetricProvider):
            raise TypeError("ngs_provider must implement NgsMetricProvider.")
        ngs_used = np.asarray(self.ngs_used, dtype=bool).reshape(-1).copy()
        if int(np.count_nonzero(ngs_used)) != self.request.ngs_x.size:
            raise ValueError("ngs_used active count must match the Hybrid request.")
        ngs_used.setflags(write=False)
        object.__setattr__(self, "ngs_used", ngs_used)


@dataclass(frozen=True)
class _ActiveNgs:
    r: u.Quantity
    theta: u.Quantity
    x: u.Quantity
    y: u.Quantity
    magnitude: u.Quantity


@dataclass(frozen=True)
class _ArtifactFileCacheKey:
    path: str
    size_bytes: int
    modified_ns: int


_NGS_HO_METRIC_INTERPOLATOR_CACHE: dict[
    _ArtifactFileCacheKey, NgsHoMetricInterpolator
] = {}
_SCIENCE_HO_PSF_INTERPOLATOR_CACHE: dict[
    _ArtifactFileCacheKey, ScienceHoPsfInterpolator
] = {}


class HybridSimulation(TiptopConfigBackedSimulation):
    """AO Predict adapter over the standalone Hybrid AO PSF engine.

    AO Predict owns payload binding, per-row option and photometry resolution,
    result persistence, statistics, and diagnostic field names. Hybrid AO PSF
    owns provider evaluation, MASTSEL execution, Ctot handling, finite-field
    blur, jitter, and immutable scientific result construction.
    """

    _VERSION = "0.0.1"

    EXTRA_STAT_JITTER = "jitter"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH = "science_ho_psf_interpolator_path"
    KEY_NGS_HO_METRIC_INTERPOLATOR_PATH = "ngs_ho_metric_interpolator_path"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_PROVENANCE = "science_ho_psf_interpolator_provenance"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_BUILDER = "science_ho_psf_interpolator_builder"
    KEY_NGS_HO_METRIC_INTERPOLATOR_PROVENANCE = "ngs_ho_metric_interpolator_provenance"
    KEY_NGS_HO_METRIC_INTERPOLATOR_BUILDER = "ngs_ho_metric_interpolator_builder"
    KEY_NON_PSD_POLICY = "non_psd_policy"
    KEY_RUNTIME_EFFECTIVE_PARSER = "effective_parser"
    KEY_RUNTIME_RESULT = "hybrid_result"
    KEY_RUNTIME_NGS_USED = "hybrid_ngs_used"

    @property
    def extra_stat_fields(self) -> Mapping[str, u.UnitBase]:
        """Return Hybrid extra-stat units persisted under ``/stats``."""
        return {self.EXTRA_STAT_JITTER: u.mas}

    @property
    def supported_extra_diagnostics_levels(self) -> tuple[str, ...]:
        """Return non-``none`` diagnostics levels implemented by Hybrid."""
        return (schema.DIAGNOSTICS_LEVEL_VALIDATION, schema.DIAGNOSTICS_LEVEL_DEBUG)

    def __init__(self) -> None:
        """Initialize unbound Hybrid simulation state."""
        super().__init__()
        self._science_ho_psf_interpolator_path: Path | None = None
        self._ngs_ho_metric_interpolator_path: Path | None = None
        self._ngs_ho_metric_interpolator: NgsHoMetricInterpolator | None = None
        self._science_ho_psf_interpolator: ScienceHoPsfInterpolator | None = None
        self._non_psd_policy = NonPsdPolicy.ERROR

    @property
    def science_ho_psf_interpolator(self) -> ScienceHoPsfInterpolator:
        """Return the configured science-HO-PSF interpolator artifact."""
        return self._get_science_ho_psf_interpolator()

    @property
    def ngs_ho_metric_interpolator(self) -> NgsHoMetricInterpolator:
        """Return the configured NGS-HO-metric interpolator artifact."""
        return self._get_ngs_ho_metric_interpolator()

    def _get_science_ho_psf_interpolator(self) -> ScienceHoPsfInterpolator:
        """Return the process-cached upstream science provider."""
        if self._science_ho_psf_interpolator is None:
            if self._science_ho_psf_interpolator_path is None:
                raise TypeError("HybridSimulation science-HO-PSF interpolator path is not configured.")
            self._science_ho_psf_interpolator = _get_cached_science_ho_psf_interpolator(
                self._science_ho_psf_interpolator_path
            )
        return self._science_ho_psf_interpolator

    def _get_ngs_ho_metric_interpolator(self) -> NgsHoMetricInterpolator:
        """Return the cached NGS-HO-metric interpolator for runtime use."""
        if self._ngs_ho_metric_interpolator is None:
            if self._ngs_ho_metric_interpolator_path is None:
                raise TypeError("HybridSimulation NGS-HO-metric interpolator path is not configured.")
            self._ngs_ho_metric_interpolator = _get_cached_ngs_ho_metric_interpolator(
                self._ngs_ho_metric_interpolator_path
            )
        return self._ngs_ho_metric_interpolator

    # Simulation payload lifecycle

    def prepare_simulation_payload(
        self,
        base_simulation_payload: Mapping[str, Any],
        simulation_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build the persisted Hybrid ``/simulation`` payload.

        The MASTSEL base INI is persisted through the inherited
        ``base_config`` lifecycle. Interpolator artifact paths are resolved
        relative to ``simulation.base_path``, loaded, validated, and persisted
        as resolved strings under ``/simulation``. The payload also records the
        compact interpolator provenance and builder identity so datasets retain
        static Hybrid input identity independently of optional per-run
        ``/diagnostics``. The exact MASTSEL base INI text remains available as
        inherited ``/simulation/base_config`` provenance.

        Args:
            base_simulation_payload: Core simulation payload containing name,
                version, and declared extra stats.
            simulation_cfg: User-facing simulation config. Hybrid requires
                ``config_path``, ``science_ho_psf_interpolator_path``, and
                ``ngs_ho_metric_interpolator_path``.

        Returns:
            Persisted ``/simulation`` mapping for Hybrid datasets.

        Raises:
            TypeError: If required path fields have invalid types.
            ValueError: If the MASTSEL INI text or interpolator artifacts are
                invalid, or if the selected diagnostics level is unsupported.
            FileNotFoundError: If required config or artifact paths are missing.
        """
        simulation_cfg = dict(simulation_cfg)
        simulation_cfg[self.KEY_NON_PSD_POLICY] = _normalize_non_psd_policy(
            simulation_cfg.get(
                self.KEY_NON_PSD_POLICY,
                NonPsdPolicy.ERROR.value,
            )
        ).value
        simulation_payload = dict(
            super().prepare_simulation_payload(
                base_simulation_payload,
                simulation_cfg,
            )
        )
        science_path = self._resolve_required_artifact_path(simulation_cfg, self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH)
        ngs_path = self._resolve_required_artifact_path(simulation_cfg, self.KEY_NGS_HO_METRIC_INTERPOLATOR_PATH)
        science = load_science_ho_psf_interpolator(science_path)
        ngs = load_ngs_ho_metric_interpolator(ngs_path)
        validate_science_ho_psf_interpolator(science)
        validate_ngs_ho_metric_interpolator(ngs)
        simulation_payload[self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH] = str(science_path)
        simulation_payload[self.KEY_NGS_HO_METRIC_INTERPOLATOR_PATH] = str(ngs_path)
        simulation_payload[self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PROVENANCE] = np.asarray(science.provenance, dtype=str)
        simulation_payload[self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_BUILDER] = dict(science.builder)
        simulation_payload[self.KEY_NGS_HO_METRIC_INTERPOLATOR_PROVENANCE] = np.asarray(ngs.provenance, dtype=str)
        simulation_payload[self.KEY_NGS_HO_METRIC_INTERPOLATOR_BUILDER] = dict(ngs.builder)
        meta_fields = _science_meta_fields(science)
        if meta_fields:
            simulation_payload[schema.KEY_SIMULATION_META_FIELDS] = {
                name: unit_string(unit) for name, unit in meta_fields.items()
            }
        return simulation_payload

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate persisted Hybrid simulation payload without binding state.

        Args:
            simulation_payload: Candidate persisted ``/simulation`` payload.

        Raises:
            TypeError: If payload fields have invalid types.
            ValueError: If the base INI or required artifact path fields are invalid.
            FileNotFoundError: If a persisted interpolator artifact path is
                missing.
        """
        super().validate_simulation_payload(simulation_payload)
        _non_psd_policy_from_payload(simulation_payload)
        science_path = self._get_required_payload_path(simulation_payload, self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH)
        ngs_path = self._get_required_payload_path(simulation_payload, self.KEY_NGS_HO_METRIC_INTERPOLATOR_PATH)
        science = load_science_ho_psf_interpolator(science_path)
        validate_science_ho_psf_interpolator(science)
        _validate_science_meta_fields_match(simulation_payload, science)
        validate_ngs_ho_metric_interpolator(load_ngs_ho_metric_interpolator(ngs_path))

    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Bind base INI plus science and NGS artifact paths.

        Binding is atomic with respect to validation of payload-owned state:
        the base INI and required artifact paths are validated before instance
        state is updated. Interpolator artifacts are loaded lazily by runtime
        getters near their use sites.

        Args:
            simulation_payload: Persisted ``/simulation`` payload read from
                dataset storage.

        Raises:
            TypeError: If payload fields have invalid types.
            ValueError: If the base INI or interpolator artifacts are invalid.
            FileNotFoundError: If a persisted interpolator artifact path is
                missing.
        """
        base_config_text = self._get_required_base_config_text(simulation_payload)
        base_config = self._prepare_base_config_binding(base_config_text)
        non_psd_policy = _non_psd_policy_from_payload(simulation_payload)
        science_path = self._get_required_payload_path(simulation_payload, self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH)
        ngs_path = self._get_required_payload_path(simulation_payload, self.KEY_NGS_HO_METRIC_INTERPOLATOR_PATH)
        self._load_base_simulation_payload(simulation_payload)
        self._base_config_text = base_config_text
        self._bind_base_config(base_config)
        self._science_ho_psf_interpolator_path = science_path
        self._ngs_ho_metric_interpolator_path = ngs_path
        self._science_ho_psf_interpolator = None
        self._ngs_ho_metric_interpolator = None
        self._non_psd_policy = non_psd_policy

    def _config_file_description(self) -> str:
        """Return the user-facing description for missing MASTSEL INI errors."""
        return "MASTSEL INI file"

    # Setup payload lifecycle

    def _parse_setup_payload(self, setup_payload: Mapping[str, Any]) -> HybridSetup:
        """Parse and validate Hybrid persisted ``/setup`` without binding it.

        Args:
            setup_payload: Candidate persisted setup payload.

        Returns:
            Parsed ``HybridSetup`` instance ready to be bound by
            ``load_setup_payload()``.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        return self._parse_base_setup_payload(setup_payload, HybridSetup)

    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Bind typed Hybrid setup from persisted ``/setup`` payload.

        Args:
            setup_payload: Persisted setup payload read from dataset storage.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        self._load_base_setup_payload(setup_payload, HybridSetup)

    # Runtime lifecycle

    def _create_runtime_context(self, index: int, options: dict[str, Any], setup: SimulationSetup) -> dict[str, Any]:
        """Create runtime scratch state for one Hybrid simulation.

        This resolves AO-owned active-star, atmosphere, and photometry inputs
        without evaluating providers or invoking the upstream engine.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Bound typed setup object for this simulation instance.

        Returns:
            Runtime scratch mapping containing the effective MASTSEL parser and
            active NGS coordinates/magnitudes.

        Raises:
            TypeError: If ``setup`` is not ``HybridSetup``.
            ValueError: If required runtime options are missing, malformed, or
                outside interpolator support.
        """
        del index
        if not isinstance(setup, HybridSetup):
            raise TypeError("HybridSimulation setup must be HybridSetup.")
        active_ngs = self._active_ngs_from_options(options)
        parser = self._build_runtime_mastsel_parser(setup, options)
        wavelength = _require_option_scalar(options, schema.KEY_OPTION_WAVELENGTH)
        zenith_angle = _require_option_scalar(options, schema.KEY_OPTION_ZENITH_ANGLE)
        validate_science_ho_psf_query(
            self.science_ho_psf_interpolator,
            zenith_angle=zenith_angle,
            wavelength=wavelength,
        )
        validate_ngs_ho_metric_query(
            self._get_ngs_ho_metric_interpolator(),
            zenith_angle=zenith_angle,
            x=active_ngs.x,
            y=active_ngs.y,
        )
        return {
            self.KEY_RUNTIME_EFFECTIVE_PARSER: parser,
            "active_ngs": active_ngs,
        }

    def run(self, context: SimulationContext) -> None:
        """Execute one AO option row through Hybrid AO PSF exactly once.

        Args:
            context: Simulation context produced by ``create()``.

        Raises:
            TypeError: If ``context.setup`` has the wrong concrete type.
            ValueError: If resolved request values or upstream results are
                invalid.
            RuntimeError: If the upstream MASTSEL execution fails.
        """
        resolved = self._resolve_hybrid_inputs(context)
        result = simulate(
            resolved.request,
            resolved.science_provider,
            resolved.ngs_provider,
        )
        context.runtime[self.KEY_RUNTIME_RESULT] = result
        context.runtime[self.KEY_RUNTIME_NGS_USED] = resolved.ngs_used

    def _resolve_hybrid_inputs(
        self,
        context: SimulationContext,
    ) -> HybridResolvedInputs:
        """Resolve the one protected downstream extension bundle.

        Subclasses may call this implementation and return a replaced request
        or wrapped provider for instrument-owned policy. They must not execute
        the engine or reinterpret its result. This method borrows the loaded
        immutable providers and does not mutate persisted setup or options.

        Args:
            context: Runtime context produced by ``create()``.

        Returns:
            Atomic upstream request, provider, and AO NGS-slot mapping bundle.

        Raises:
            TypeError: If the bound setup or runtime scratch state is invalid.
            ValueError: If required option or photometry values are invalid.
        """
        setup = self._resolved_science_setup(context)
        if not isinstance(setup, HybridSetup):
            raise TypeError("context.setup must be HybridSetup.")
        options = context.options
        active_ngs = context.runtime.get("active_ngs")
        if not isinstance(active_ngs, _ActiveNgs):
            active_ngs = self._active_ngs_from_options(options)
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")
        science_x, science_y = polar_to_cartesian(setup.sci_r, setup.sci_theta)
        ngs_flux = self._ngs_flux_from_config(
            parser,
            active_ngs.magnitude,
            setup,
        )
        ngs_frame_rate = np.full(
            active_ngs.magnitude.size,
            self._get_frame_rate_lo(parser),
            dtype=float,
        ) * u.Hz
        return HybridResolvedInputs(
            request=HybridRequest(
                science_x=science_x,
                science_y=science_y,
                ngs_x=active_ngs.x,
                ngs_y=active_ngs.y,
                wavelength=_require_option_scalar(
                    options,
                    schema.KEY_OPTION_WAVELENGTH,
                ),
                zenith_angle=_require_option_scalar(
                    options,
                    schema.KEY_OPTION_ZENITH_ANGLE,
                ),
                ngs_flux=ngs_flux,
                ngs_frame_rate=ngs_frame_rate,
                mastsel_ini=_serialize_parser(parser),
                non_psd_policy=self._non_psd_policy,
                diagnostics_level=DiagnosticsLevel(self.diagnostics_level),
            ),
            science_provider=self._get_science_ho_psf_interpolator(),
            ngs_provider=self._get_ngs_ho_metric_interpolator(),
            ngs_used=_ngs_used_mask(options),
        )

    def _extract_psfs(self, context: SimulationContext) -> np.ndarray | None:
        """Extract the Hybrid PSF cube from runtime state.

        Args:
            context: Completed simulation context.

        Returns:
            PSF cube with shape ``[M, Ny, Nx]``.

        Raises:
            ValueError: If Hybrid runtime did not produce a result.
        """
        return np.asarray(_require_hybrid_result(context).psfs, dtype=np.float32)

    def prepare_psfs_for_stats(
        self,
        psfs: np.ndarray,
        setup: Mapping[str, Any] | SimulationSetup,
        meta: Mapping[str, Any],
    ) -> np.ndarray:
        """Validate metric-ready Hybrid PSFs without renormalizing them.

        Hybrid science-HO PSFs are prepared in their source flux convention
        before interpolation, and finite-FOV Ctot blur can move energy outside
        the retained image support. Core-stat preprocessing must therefore not
        clip-and-sum-normalize the result after runtime blur.
        """
        del setup, meta
        psfs_array = np.asarray(psfs)
        if psfs_array.ndim != 3:
            raise ValueError(f"Hybrid PSFs must have shape (N, y, x); got {psfs_array.shape}.")
        if not np.all(np.isfinite(psfs_array)):
            raise ValueError("Hybrid PSFs must contain only finite values.")
        if np.any(psfs_array < 0.0):
            raise ValueError("Hybrid PSFs must be non-negative.")
        flux = np.sum(psfs_array, axis=(-2, -1), dtype=np.float64)
        if not np.all(np.isfinite(flux)) or np.any(flux <= 0.0):
            raise ValueError("Hybrid PSFs must have positive finite flux.")
        return psfs

    def _extract_psf_parameters(self, context: SimulationContext) -> PsfParameters:
        """Extract provider-backed PSF metadata from runtime state.

        Args:
            context: Completed simulation context.

        Returns:
            Pixel scale, telescope diameter, and telescope pupil associated
            with the Hybrid provider PSFs.

        Raises:
            ValueError: If Hybrid runtime did not produce a result.
        """
        result = _require_hybrid_result(context)
        return PsfParameters(
            pixel_scale=result.metadata.pixel_scale,
            tel_diameter=result.metadata.tel_diameter,
            tel_pupil=result.metadata.tel_pupil,
        )

    def finalize(self, context: SimulationContext) -> None:
        """Finalize Hybrid output and attach provider source metadata.

        Hybrid providers may supply evaluated scalar metadata from their
        science-HO-PSF source artifacts. These fields are declared in
        ``/simulation/meta_fields`` and copied into ``SimulationResult.meta``
        before the runner computes core PSF statistics.
        """
        super().finalize(context)
        if context.result is None:
            raise ValueError("Hybrid finalize did not produce a simulation result.")
        context.result.meta.update(
            dict(_require_hybrid_result(context).source_metadata)
        )

    def build_extra_stats(self, context: SimulationContext) -> Mapping[str, Any]:
        """Return Hybrid jitter from MASTSEL Ctot.

        Args:
            context: Completed simulation context.

        Returns:
            Mapping containing the declared ``jitter`` extra stat in
            milliarcseconds, one value per science point.

        Raises:
            ValueError: If Hybrid runtime did not produce a result.
        """
        return {self.EXTRA_STAT_JITTER: _require_hybrid_result(context).jitter}

    def _diagnostic_field_specs(self, diagnostics_level: str) -> Mapping[str, Mapping[str, Any]]:
        """Return persisted Hybrid diagnostic field specs for one level.

        Specs are keyed by slash-delimited ``/diagnostics`` paths and contain a
        storage ``dtype`` plus a shape excluding the leading simulation
        dimension ``N``. The supported non-``none`` levels are ``validation``
        and ``debug``. ``debug`` includes all validation diagnostics plus full
        Ctot cubes and runtime INI text.

        Args:
            diagnostics_level: Requested non-``none`` diagnostics level.

        Returns:
            Mapping from slash-delimited diagnostic field path to dtype/shape
            spec. Shape tokens use the generic diagnostics vocabulary
            ``num_sci`` and ``num_ngs``.

        Raises:
            ValueError: If subclass extension specs collide with upstream
                Hybrid diagnostic fields.
        """
        specs: dict[str, Mapping[str, Any]] = {
            "hybrid/angle_to_wavefront_scale": {
                "dtype": "float32",
                "shape": (),
                "unit": u.nm / u.mas,
            },
            "hybrid/psd_valid_count": {"dtype": "int32", "shape": ()},
            "hybrid/psd_valid_fraction": {"dtype": "float32", "shape": (), "unit": u.dimensionless_unscaled},
            "hybrid/psd_valid_mask": {"dtype": "bool", "shape": ("num_sci",)},
            "hybrid/psd_clipped": {"dtype": "bool", "shape": ("num_sci",)},
            "hybrid/ctot_angle_trace_min": {"dtype": "float32", "shape": (), "unit": u.mas**2},
            "hybrid/ctot_angle_trace_max": {"dtype": "float32", "shape": (), "unit": u.mas**2},
            "hybrid/ctot_angle_trace_mean": {"dtype": "float32", "shape": (), "unit": u.mas**2},
            "hybrid/ctot_angle_determinant_min": {"dtype": "float32", "shape": (), "unit": u.mas**4},
            "hybrid/ctot_angle_determinant_max": {"dtype": "float32", "shape": (), "unit": u.mas**4},
            "hybrid/ctot_angle_determinant_mean": {"dtype": "float32", "shape": (), "unit": u.mas**4},
            "hybrid/ngs_used": {"dtype": "bool", "shape": ("num_ngs",)},
            "hybrid/ngs/ee": {"dtype": "float32", "shape": ("num_ngs",), "unit": u.dimensionless_unscaled},
            "hybrid/ngs/fwhm": {"dtype": "float32", "shape": ("num_ngs",), "unit": u.mas},
            "hybrid/ngs/sr": {"dtype": "float32", "shape": ("num_ngs",), "unit": u.dimensionless_unscaled},
            "hybrid/ngs/flux": {"dtype": "float32", "shape": ("num_ngs",), "unit": u.photon / u.s},
            "hybrid/ngs/frequency": {"dtype": "float32", "shape": ("num_ngs",), "unit": u.Hz},
        }
        if diagnostics_level == schema.DIAGNOSTICS_LEVEL_DEBUG:
            specs.update(
                {
                    "hybrid/ctot_wavefront": {"dtype": "float32", "shape": ("num_sci", 2, 2), "unit": u.nm**2},
                    "hybrid/ctot_angle": {"dtype": "float32", "shape": ("num_sci", 2, 2), "unit": u.mas**2},
                    "hybrid/runtime_ini_text": {"dtype": "str", "shape": ()},
                }
            )
        extension = dict(self._extend_hybrid_diagnostic_field_specs(diagnostics_level))
        collisions = sorted(set(specs) & set(extension))
        if collisions:
            raise ValueError(f"Hybrid diagnostic extension fields collide with upstream fields: {', '.join(collisions)}")
        specs.update(extension)
        return specs

    def _extend_hybrid_diagnostic_field_specs(self, diagnostics_level: str) -> Mapping[str, Mapping[str, Any]]:
        """Return subclass-specific Hybrid diagnostic field specs.

        Subclasses may declare additional slash-delimited fields persisted
        under ``/diagnostics`` for the selected level. Returned specs must use
        the same schema as upstream specs: a ``dtype`` and a shape excluding the
        leading simulation dimension ``N``. This hook must not mutate instance
        state and must not return keys owned by upstream Hybrid diagnostics.

        Args:
            diagnostics_level: Bound diagnostics level for this run family.

        Returns:
            Additional field specs keyed by slash-delimited diagnostics paths.
            Keys must not collide with upstream Hybrid diagnostic fields.
        """
        del diagnostics_level
        return {}

    def _extract_diagnostics(self, context: SimulationContext) -> Mapping[str, Any]:
        """Extract optional Hybrid diagnostics from completed runtime state.

        The base Hybrid implementation returns no diagnostics when
        ``diagnostics_level`` is ``none``. For ``validation`` and ``debug`` it
        returns values matching the field specs declared by
        ``_diagnostic_field_specs()`` and merges subclass extension values after
        checking that extension keys do not overwrite upstream keys.

        Args:
            context: Completed simulation context whose runtime contains a
                Hybrid result.

        Returns:
            Flat mapping keyed by slash-delimited ``/diagnostics`` paths.

        Raises:
            ValueError: If runtime state is incomplete or extension diagnostics
                collide with upstream fields.
        """
        if self.diagnostics_level == schema.DIAGNOSTICS_LEVEL_NONE:
            return {}
        diagnostics_context = HybridDiagnosticsContext(
            result=_require_hybrid_result(context),
            ngs_used=_require_hybrid_ngs_used(context),
        )
        diagnostics = dict(self._build_hybrid_diagnostics(diagnostics_context))
        extension = dict(self._extend_hybrid_diagnostics(diagnostics_context))
        collisions = sorted(set(diagnostics) & set(extension))
        if collisions:
            raise ValueError(f"Hybrid diagnostic extension fields collide with upstream fields: {', '.join(collisions)}")
        diagnostics.update(extension)
        return diagnostics

    def _extend_hybrid_diagnostics(self, diagnostics_context: HybridDiagnosticsContext) -> Mapping[str, Any]:
        """Return subclass-specific diagnostics for a completed Hybrid run.

        Subclasses should return only values declared by
        ``_extend_hybrid_diagnostic_field_specs()`` for the active diagnostics
        level. Keys are slash-delimited ``/diagnostics`` paths and values must
        match the declared dtype and per-simulation shape. This hook receives a
        narrow immutable diagnostics context and must not mutate Hybrid runtime
        state.

        Args:
            diagnostics_context: Runtime values already produced by the generic
                Hybrid execution path.

        Returns:
            Additional per-run diagnostic values. The default returns no
            subclass diagnostics.
        """
        del diagnostics_context
        return {}

    def _build_hybrid_diagnostics(self, diagnostics_context: HybridDiagnosticsContext) -> Mapping[str, Any]:
        """Map upstream Hybrid diagnostics to AO persisted field names.

        ``validation`` diagnostics include compact scalar Ctot summaries, PSD
        validity and clipping masks, slot-aligned active-NGS mask, and slot-aligned
        MASTSEL inputs. ``debug`` adds full Ctot cubes and serialized runtime
        INI text. Returned values are flat and must match the specs from
        ``_diagnostic_field_specs()`` for the active level.

        Args:
            diagnostics_context: Runtime values produced by one completed
                Hybrid simulation.

        Returns:
            Upstream Hybrid diagnostic values keyed by slash-delimited
            ``/diagnostics`` paths.

        Raises:
            ValueError: If required MASTSEL input diagnostics are missing,
                non-finite, or inconsistent with the active NGS mask.
        """
        result = diagnostics_context.result
        upstream = result.diagnostics
        if upstream is None:
            raise ValueError(
                "Hybrid AO PSF did not return the requested diagnostics."
            )
        used = np.asarray(diagnostics_context.ngs_used, dtype=bool)
        diagnostics: dict[str, Any] = {
            "hybrid/angle_to_wavefront_scale": np.float32(
                result.angle_to_wavefront_scale.to_value(u.nm / u.mas)
            )
            * (u.nm / u.mas),
            "hybrid/psd_valid_count": np.int32(upstream.psd_valid_count),
            "hybrid/psd_valid_fraction": np.float32(
                upstream.psd_valid_fraction
            )
            * u.dimensionless_unscaled,
            "hybrid/psd_valid_mask": np.asarray(
                upstream.psd_valid_mask,
                dtype=bool,
            ),
            "hybrid/psd_clipped": np.asarray(result.psd_clipped, dtype=bool),
            "hybrid/ctot_angle_trace_min": upstream.ctot_angle_trace_min.astype(
                np.float32
            ),
            "hybrid/ctot_angle_trace_max": upstream.ctot_angle_trace_max.astype(
                np.float32
            ),
            "hybrid/ctot_angle_trace_mean": upstream.ctot_angle_trace_mean.astype(
                np.float32
            ),
            "hybrid/ctot_angle_determinant_min": upstream.ctot_angle_determinant_min.astype(
                np.float32
            ),
            "hybrid/ctot_angle_determinant_max": upstream.ctot_angle_determinant_max.astype(
                np.float32
            ),
            "hybrid/ctot_angle_determinant_mean": upstream.ctot_angle_determinant_mean.astype(
                np.float32
            ),
            "hybrid/ngs_used": used,
            "hybrid/ngs/ee": _slot_aligned(result.ngs_metrics.ee, used),
            "hybrid/ngs/fwhm": _slot_aligned(result.ngs_metrics.fwhm, used),
            "hybrid/ngs/sr": _slot_aligned(result.ngs_metrics.sr, used),
            "hybrid/ngs/flux": _slot_aligned(result.ngs_flux, used),
            "hybrid/ngs/frequency": _slot_aligned(result.ngs_frame_rate, used),
        }
        if self.diagnostics_level == schema.DIAGNOSTICS_LEVEL_DEBUG:
            diagnostics.update(
                {
                    "hybrid/ctot_wavefront": result.ctot_wavefront.astype(
                        np.float32
                    ),
                    "hybrid/ctot_angle": result.ctot_angle.astype(np.float32),
                    "hybrid/runtime_ini_text": result.effective_mastsel_ini,
                }
            )
        return diagnostics

    def _build_runtime_mastsel_parser(
        self,
        setup: HybridSetup,
        options: Mapping[str, Any],
    ) -> ConfigParser:
        """Build AO-owned atmosphere overrides for the upstream MASTSEL INI."""
        parser = deepcopy(self.base_config.parser)
        self._update_atmosphere_in_ini(parser, options, setup)
        return parser

    def _update_atmosphere_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: HybridSetup,
    ) -> None:
        """Apply atmosphere/profile/r0 runtime updates for MASTSEL.

        MASTSEL accepts either ``atmosphere.r0_Value`` or
        ``atmosphere.Seeing``. If both are present, MASTSEL discards
        ``r0_Value`` and uses ``Seeing``. Hybrid writes only ``r0_Value`` so
        the AO Predict runtime ``r0`` reaches MASTSEL directly.
        """
        if not parser.has_section("atmosphere"):
            return
        atmosphere = parser["atmosphere"]
        profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
        profile = atm.select_atm_profile(setup.atm_profiles, profile_id)
        self._write_atmosphere_profile_fields(parser, profile)

        r0 = _runtime_r0_m(setup, options)
        atmosphere["r0_Value"] = f"{r0.to_value(u.m):.6g}"
        if "Seeing" in atmosphere:
            del atmosphere["Seeing"]

    def _ngs_flux_from_config(self, parser: ConfigParser, ngs_magnitude: u.Quantity, setup: HybridSetup) -> u.Quantity:
        """Return active NGS flux in photons per second for MASTSEL."""
        photometry = self._get_ngs_photometry_config(parser, setup.ngs_magnitude_zeropoint)
        photons_per_frame = magnitudes_to_photons_per_frame(ngs_magnitude, photometry)
        return photons_per_frame.to_value(u.photon) * photometry.frame_rate.to_value(u.Hz) * (u.photon / u.s)

    # Path and option helpers

    def _resolve_required_artifact_path(self, simulation_cfg: Mapping[str, Any], key: str) -> Path:
        """Resolve a required simulation artifact path through ``base_path``."""
        raw_path = simulation_cfg.get(key)
        if raw_path is None:
            raise ValueError(f"HybridSimulation requires simulation['{key}'].")
        if not isinstance(raw_path, str):
            raise TypeError(f"simulation['{key}'] must be a string.")
        path = Path(raw_path)
        if not path.is_absolute():
            base_path = simulation_cfg.get(schema.KEY_CFG_SIMULATION_BASE_PATH)
            if base_path is not None:
                if not isinstance(base_path, str):
                    raise TypeError(f"simulation['{schema.KEY_CFG_SIMULATION_BASE_PATH}'] must be a string when provided.")
                path = Path(base_path) / path
        if not path.is_file():
            raise FileNotFoundError(f"HybridSimulation artifact not found for simulation['{key}']: {path}")
        return path.resolve()

    def _get_required_payload_path(self, simulation_payload: Mapping[str, Any], key: str) -> Path:
        """Read a required persisted artifact path from ``/simulation``."""
        if key not in simulation_payload:
            raise ValueError(f"HybridSimulation requires simulation['{key}'].")
        raw_path = simulation_payload[key]
        if not isinstance(raw_path, str):
            raise TypeError(f"simulation['{key}'] must be a string for HybridSimulation.")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"HybridSimulation persisted artifact path not found for simulation['{key}']: {path}")
        return path.resolve()

    def _active_ngs_from_options(self, options: Mapping[str, Any]) -> _ActiveNgs:
        """Return active NGS coordinate and magnitude vectors from runtime options."""
        for key in (schema.KEY_OPTION_NGS_R, schema.KEY_OPTION_NGS_THETA, schema.KEY_OPTION_NGS_MAGNITUDE):
            if key not in options:
                raise ValueError(f"HybridSimulation options require '{key}'.")
        if schema.KEY_OPTION_NGS_USED not in options:
            raise ValueError(
                "Missing required runtime option 'ngs_used'. Call runner.prepare_options_payload(...) "
                "or api.init_dataset(...) so core derives runtime fields first."
            )
        r = quantity_value(options[schema.KEY_OPTION_NGS_R], u.arcsec, label=schema.KEY_OPTION_NGS_R, dtype=float).reshape(-1)
        theta = quantity_value(options[schema.KEY_OPTION_NGS_THETA], u.deg, label=schema.KEY_OPTION_NGS_THETA, dtype=float).reshape(-1)
        magnitude = quantity_value(options[schema.KEY_OPTION_NGS_MAGNITUDE], u.mag, label=schema.KEY_OPTION_NGS_MAGNITUDE, dtype=float).reshape(-1)
        used = np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)
        if r.shape != theta.shape or r.shape != magnitude.shape or r.shape != used.shape:
            raise ValueError("HybridSimulation NGS option vectors and ngs_used must have identical shape.")
        if not np.any(used):
            raise ValueError("HybridSimulation requires at least one active NGS.")
        active_r = r[used]
        active_theta = theta[used]
        active_magnitude = magnitude[used]
        if not np.all(np.isfinite(active_r)) or not np.all(np.isfinite(active_theta)) or not np.all(np.isfinite(active_magnitude)):
            raise ValueError("HybridSimulation active NGS options must be finite.")
        active_r_quantity = active_r * u.arcsec
        active_theta_quantity = active_theta * u.deg
        x, y = polar_to_cartesian(active_r_quantity, active_theta_quantity)
        return _ActiveNgs(
            r=active_r_quantity,
            theta=active_theta_quantity,
            x=x,
            y=y,
            magnitude=active_magnitude * u.mag,
        )


def _require_hybrid_result(context: SimulationContext) -> HybridResult:
    result = context.runtime.get(HybridSimulation.KEY_RUNTIME_RESULT)
    if not isinstance(result, HybridResult):
        raise ValueError(  # noqa: TRY004 - incomplete runtime state is a lifecycle error
            "Missing Hybrid runtime result. Did run(...) complete?"
        )
    return result


def _require_hybrid_ngs_used(context: SimulationContext) -> np.ndarray:
    value = context.runtime.get(HybridSimulation.KEY_RUNTIME_NGS_USED)
    if value is None:
        raise ValueError("Missing Hybrid NGS slot mapping. Did run(...) complete?")
    return np.asarray(value, dtype=bool).reshape(-1)


def _science_meta_fields(interpolator: ScienceHoPsfInterpolator) -> dict[str, u.UnitBase]:
    return {str(name): value.unit for name, value in interpolator.meta.items()}


def _validate_science_meta_fields_match(
    simulation_payload: Mapping[str, Any],
    interpolator: ScienceHoPsfInterpolator,
) -> None:
    payload_fields = normalize_meta_fields(simulation_payload.get(schema.KEY_SIMULATION_META_FIELDS, {}))
    artifact_fields = {
        name: unit_string(unit)
        for name, unit in _science_meta_fields(interpolator).items()
    }
    if payload_fields != artifact_fields:
        raise ValueError(
            "HybridSimulation science meta field registry mismatch: "
            f"payload has {list(payload_fields)}, artifact has {list(artifact_fields)}."
        )


def _runtime_r0_m(setup: HybridSetup, options: Mapping[str, Any]) -> u.Quantity:
    if schema.KEY_OPTION_R0 in options:
        return _require_option_scalar(options, schema.KEY_OPTION_R0)
    profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
    profile = atm.select_atm_profile(setup.atm_profiles, profile_id)
    return profile[atm.KEY_SETUP_ATM_PROFILE_R0].to(u.m)


def _ngs_used_mask(options: Mapping[str, Any]) -> np.ndarray:
    """Return the runtime NGS slot mask."""
    if schema.KEY_OPTION_NGS_USED not in options:
        raise ValueError("Hybrid diagnostics require runtime option 'ngs_used'.")
    return np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)


def _slot_aligned(active_values: u.Quantity, used: np.ndarray) -> u.Quantity:
    """Return NGS slot-aligned values with inactive slots set to ``NaN``."""
    active = np.asarray(active_values.value, dtype=float).reshape(-1)
    used = np.asarray(used, dtype=bool).reshape(-1)
    if active.shape[0] != int(np.count_nonzero(used)):
        raise ValueError("Hybrid diagnostics active NGS values do not match ngs_used.")
    out = np.full(used.shape, np.nan, dtype=np.float32)
    out[used] = active.astype(np.float32)
    return out * active_values.unit


def _require_option_scalar(options: Mapping[str, Any], key: str) -> u.Quantity:
    if key not in options:
        raise ValueError(f"HybridSimulation options require '{key}'.")
    unit = schema.OPTION_FIELD_UNITS[key]
    values = quantity_value(options[key], unit, label=f"options.{key}", dtype=float)
    if values.ndim != 0:
        raise ValueError(f"HybridSimulation option '{key}' must be scalar.")
    value = float(values.item())
    if not np.isfinite(value):
        raise ValueError(f"HybridSimulation option '{key}' must be finite.")
    return value * unit


def _normalize_non_psd_policy(value: Any) -> NonPsdPolicy:
    try:
        return NonPsdPolicy(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("non_psd_policy must be 'error' or 'clip'.") from exc


def _non_psd_policy_from_payload(
    simulation_payload: Mapping[str, Any],
) -> NonPsdPolicy:
    return _normalize_non_psd_policy(
        simulation_payload.get(
            HybridSimulation.KEY_NON_PSD_POLICY,
            NonPsdPolicy.ERROR.value,
        )
    )


def _artifact_file_cache_key(path: Path) -> _ArtifactFileCacheKey:
    resolved_path = Path(path).expanduser().resolve()
    stat = resolved_path.stat()
    return _ArtifactFileCacheKey(
        path=str(resolved_path),
        size_bytes=int(stat.st_size),
        modified_ns=int(stat.st_mtime_ns),
    )


def _get_cached_science_ho_psf_interpolator(
    path: Path,
) -> ScienceHoPsfInterpolator:
    key = _artifact_file_cache_key(path)
    interpolator = _SCIENCE_HO_PSF_INTERPOLATOR_CACHE.get(key)
    if interpolator is None:
        interpolator = load_science_ho_psf_interpolator(path)
        validate_science_ho_psf_interpolator(interpolator)
        _SCIENCE_HO_PSF_INTERPOLATOR_CACHE[key] = interpolator
    return interpolator


def _get_cached_ngs_ho_metric_interpolator(path: Path) -> NgsHoMetricInterpolator:
    key = _artifact_file_cache_key(path)
    interpolator = _NGS_HO_METRIC_INTERPOLATOR_CACHE.get(key)
    if interpolator is None:
        interpolator = load_ngs_ho_metric_interpolator(path)
        validate_ngs_ho_metric_interpolator(interpolator)
        _NGS_HO_METRIC_INTERPOLATOR_CACHE[key] = interpolator
    return interpolator


__all__ = [
    "HybridDiagnosticsContext",
    "HybridResolvedInputs",
    "HybridSetup",
    "HybridSimulation",
    "polar_to_cartesian",
]
