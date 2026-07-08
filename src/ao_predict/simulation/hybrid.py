"""Hybrid simulation implementation."""

from __future__ import annotations

from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from ao_predict.interpolation import (
    NgsHoMetricInterpolator,
    ScienceHoPsfInterpolator,
    evaluate_ngs_ho_metric_interpolator,
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
    validate_science_ho_psf_interpolator,
    validate_science_ho_psf_query,
)
from ao_predict.interpolation.science_ho_psf import (
    _ScienceHoPsfRuntimeInterpolator,
    _evaluate_science_ho_psf_runtime_interpolator,
    _prepare_science_ho_psf_runtime_interpolator,
)

from . import atm
from . import schema
from .base import BaseSimulationSetup, PsfParameters
from .interfaces import SimulationContext, SimulationSetup
from .photometry import magnitudes_to_photons_per_frame
from .stats import PsfMetadata
from .tiptop_config_backed import (
    TiptopConfigBackedSimulation,
    _format_ini_array,
    _serialize_parser,
)
from .validation import normalize_meta_field_names
from ..utils import as_float_vector


@dataclass(frozen=True)
class HybridSetup(BaseSimulationSetup):
    """Typed setup payload for ``HybridSimulation``.

    This setup currently adds no fields beyond ``BaseSimulationSetup``. It
    gives the upstream Hybrid runtime its own type without inheriting
    from TIPTOP runtime setup.
    """


@dataclass(frozen=True)
class SciencePsfProviderResult:
    """Science-HO PSFs and metadata returned by a Hybrid provider.

    Attributes:
        psfs: Science PSF cube with shape ``(points, y, x)``. The values remain
            in the science interpolator artifact's flux convention.
        metadata: Full PSF metadata for the predicted science PSFs. The pixel
            scale may come from the artifact-backed provider or a subclass
            override, and the flux convention must match ``psfs``.
        meta: Evaluated scalar source metadata fields that should travel with
            the simulation result under ``/meta``.
    """

    psfs: np.ndarray
    metadata: PsfMetadata
    meta: Mapping[str, float] = field(default_factory=dict)

    @property
    def pixel_scale_mas(self) -> float | np.ndarray:
        """Return the provider-supplied science PSF pixel scale."""
        return self.metadata.pixel_scale_mas

    @property
    def tel_diameter_m(self) -> float:
        """Return the provider-supplied telescope diameter."""
        return self.metadata.tel_diameter_m

    @property
    def tel_pupil(self) -> np.ndarray:
        """Return the provider-supplied telescope pupil."""
        return self.metadata.tel_pupil


@dataclass(frozen=True)
class NgsMetricProviderResult:
    """NGS-HO metrics returned by a Hybrid metric provider.

    Attributes:
        ee: Encircled-energy values, one per active NGS point.
        fwhm_mas: FWHM values in milliarcseconds, one per active NGS point.
        sr: Strehl-ratio values, one per active NGS point.
    """

    ee: np.ndarray
    fwhm_mas: np.ndarray
    sr: np.ndarray


@dataclass(frozen=True)
class HybridCtotResult:
    """MASTSEL Ctot output for one Hybrid simulation.

    Attributes:
        ctot_nm2: MASTSEL covariance cube in ``nm^2`` with shape
            ``(science_points, 2, 2)``.
        ctot_mas2: Same covariance cube converted to ``mas^2`` using the
            MASTSEL-provided ``mas2nm`` value.
        mas2nm: MASTSEL covariance conversion value used for this Ctot.
        ngs_flux: Active NGS flux values passed to MASTSEL.
        ngs_frequency: Active NGS frequency values passed to MASTSEL.
        runtime_ini_text: Serialized runtime INI text passed to MASTSEL.
    """

    ctot_nm2: np.ndarray
    ctot_mas2: np.ndarray
    mas2nm: float
    ngs_flux: np.ndarray | None = None
    ngs_frequency: np.ndarray | None = None
    runtime_ini_text: str | None = None


@dataclass(frozen=True)
class HybridDiagnosticsContext:
    """Typed runtime state available to Hybrid diagnostics hooks.

    Attributes:
        active_ngs: Active NGS coordinate and magnitude vectors.
        metrics: Active NGS-HO metrics passed to MASTSEL.
        ctot: MASTSEL Ctot result and unit conversion metadata.
        psd_mask: Per-science-point mask of PSD, nonzero Ctot matrices.
        ngs_used: Slot-aligned mask of active NGS stars for this option row.
    """

    active_ngs: "_ActiveNgs"
    metrics: NgsMetricProviderResult
    ctot: HybridCtotResult
    psd_mask: np.ndarray
    ngs_used: np.ndarray


@dataclass(frozen=True)
class _HybridRuntimeResult:
    psfs: np.ndarray
    metadata: PsfMetadata
    meta: Mapping[str, float]
    jitter_mas: np.ndarray
    diagnostics_context: HybridDiagnosticsContext


@dataclass(frozen=True)
class _ActiveNgs:
    r_arcsec: np.ndarray
    theta_deg: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    mag: np.ndarray


@dataclass(frozen=True)
class _ArtifactFileCacheKey:
    path: str
    size_bytes: int
    modified_ns: int


_NGS_HO_METRIC_INTERPOLATOR_CACHE: dict[_ArtifactFileCacheKey, NgsHoMetricInterpolator] = {}
_SCIENCE_HO_PSF_RUNTIME_INTERPOLATOR_CACHE: dict[_ArtifactFileCacheKey, _ScienceHoPsfRuntimeInterpolator] = {}


class LowOrderMas2NmAdapter:
    """Minimal low-order model adapter exposing a fixed ``mas2nm`` scale."""

    def __init__(self, mas2nm: float) -> None:
        """Initialize the adapter with a positive finite MASTSEL scale."""
        self.mas2nm = float(mas2nm)
        if not np.isfinite(self.mas2nm) or self.mas2nm <= 0.0:
            raise ValueError(f"Invalid MASTSEL mas2nm conversion: {mas2nm!r}.")


class HybridSimulation(TiptopConfigBackedSimulation):
    """MASTSEL Hybrid `Simulation` implementation.

    ``HybridSimulation`` consumes generic AO Predict science-HO-PSF and
    NGS-HO-metric interpolator artifacts plus a MASTSEL-compatible INI base
    configuration. Runtime execution predicts the science-HO PSF field,
    evaluates NGS-HO metrics, computes MASTSEL Ctot directly, applies the Ctot
    blur, and exposes ``jitter`` as the only extra persisted statistic.
    """

    _VERSION = "0.0.1"

    EXTRA_STAT_JITTER = "jitter"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH = "science_ho_psf_interpolator_path"
    KEY_NGS_HO_METRIC_INTERPOLATOR_PATH = "ngs_ho_metric_interpolator_path"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_PROVENANCE = "science_ho_psf_interpolator_provenance"
    KEY_SCIENCE_HO_PSF_INTERPOLATOR_BUILDER = "science_ho_psf_interpolator_builder"
    KEY_NGS_HO_METRIC_INTERPOLATOR_PROVENANCE = "ngs_ho_metric_interpolator_provenance"
    KEY_NGS_HO_METRIC_INTERPOLATOR_BUILDER = "ngs_ho_metric_interpolator_builder"
    KEY_RUNTIME_EFFECTIVE_PARSER = "effective_parser"
    KEY_RUNTIME_RESULT = "hybrid_result"

    @property
    def extra_stat_names(self) -> tuple[str, ...]:
        """Return Hybrid extra stat names persisted under ``/stats``."""
        return (self.EXTRA_STAT_JITTER,)

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
        self._science_ho_psf_runtime_interpolator: _ScienceHoPsfRuntimeInterpolator | None = None

    @property
    def science_ho_psf_interpolator(self) -> ScienceHoPsfInterpolator:
        """Return the configured science-HO-PSF interpolator artifact."""
        return self._get_science_ho_psf_runtime_interpolator().artifact

    @property
    def ngs_ho_metric_interpolator(self) -> NgsHoMetricInterpolator:
        """Return the configured NGS-HO-metric interpolator artifact."""
        return self._get_ngs_ho_metric_interpolator()

    def _get_science_ho_psf_runtime_interpolator(self) -> _ScienceHoPsfRuntimeInterpolator:
        """Return the cached science-HO-PSF runtime interpolator."""
        if self._science_ho_psf_runtime_interpolator is None:
            if self._science_ho_psf_interpolator_path is None:
                raise TypeError("HybridSimulation science-HO-PSF interpolator path is not configured.")
            self._science_ho_psf_runtime_interpolator = _get_cached_science_ho_psf_runtime_interpolator(
                self._science_ho_psf_interpolator_path
            )
        return self._science_ho_psf_runtime_interpolator

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
        simulation_payload = dict(super().prepare_simulation_payload(base_simulation_payload, simulation_cfg))
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
        meta_fields = _science_meta_field_names(science)
        if meta_fields:
            simulation_payload[schema.KEY_SIMULATION_META_FIELDS] = np.asarray(meta_fields, dtype=str)
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
        science_path = self._get_required_payload_path(simulation_payload, self.KEY_SCIENCE_HO_PSF_INTERPOLATOR_PATH)
        ngs_path = self._get_required_payload_path(simulation_payload, self.KEY_NGS_HO_METRIC_INTERPOLATOR_PATH)
        self._load_base_simulation_payload(simulation_payload)
        self._base_config_text = base_config_text
        self._bind_base_config(base_config)
        self._science_ho_psf_interpolator_path = science_path
        self._ngs_ho_metric_interpolator_path = ngs_path
        self._science_ho_psf_runtime_interpolator = None
        self._ngs_ho_metric_interpolator = None

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

        This derives the per-simulation effective MASTSEL INI parser by copying
        the loaded base config and applying setup- and option-dependent runtime
        overrides. It also validates the active science and NGS interpolation
        queries before any MASTSEL execution is attempted.

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
        parser = self._build_runtime_mastsel_parser(setup, options, active_ngs)
        wavelength_um = _require_option_scalar(options, schema.KEY_OPTION_WAVELENGTH_UM)
        zenith_angle_deg = _require_option_scalar(options, schema.KEY_OPTION_ZENITH_ANGLE_DEG)
        validate_science_ho_psf_query(
            self.science_ho_psf_interpolator,
            zenith_angle_deg=zenith_angle_deg,
            wavelength_um=wavelength_um,
        )
        validate_ngs_ho_metric_query(
            self._get_ngs_ho_metric_interpolator(),
            zenith_angle_deg=zenith_angle_deg,
            x_arcsec=active_ngs.x_arcsec,
            y_arcsec=active_ngs.y_arcsec,
        )
        return {
            self.KEY_RUNTIME_EFFECTIVE_PARSER: parser,
            "active_ngs": active_ngs,
        }

    def run(self, context: SimulationContext) -> None:
        """Execute one Hybrid simulation with MASTSEL Ctot blur.

        Args:
            context: Simulation context produced by ``create()``.

        Raises:
            TypeError: If ``context.setup`` has the wrong concrete type.
            ValueError: If science PSFs, NGS metrics, MASTSEL Ctot, or MASTSEL
                unit conversion values are invalid.
            RuntimeError: If MASTSEL cannot load the generated runtime INI.
        """
        setup = context.setup
        if not isinstance(setup, HybridSetup):
            raise TypeError("context.setup must be HybridSetup.")
        options = context.options
        active_ngs = context.runtime.get("active_ngs")
        if not isinstance(active_ngs, _ActiveNgs):
            active_ngs = self._active_ngs_from_options(options)
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")

        science = self._predict_science_psfs(setup, options)
        metrics = self._predict_ngs_metrics(active_ngs, options)
        ctot = self._compute_mastsel_ctot(parser, setup, active_ngs, metrics)
        psd_mask = psd_valid_mask(ctot.ctot_nm2)
        psfs = np.asarray(science.psfs, dtype=np.float32).copy()
        apply_ctot_blur(
            psfs,
            ctot.ctot_nm2,
            pixel_scale_mas=science.pixel_scale_mas,
            mas2nm=ctot.mas2nm,
        )
        context.runtime[self.KEY_RUNTIME_RESULT] = _HybridRuntimeResult(
            psfs=psfs,
            metadata=science.metadata,
            meta=dict(science.meta),
            jitter_mas=jitter_mas_from_ctot(ctot.ctot_mas2),
            diagnostics_context=HybridDiagnosticsContext(
                active_ngs=active_ngs,
                metrics=metrics,
                ctot=ctot,
                psd_mask=psd_mask,
                ngs_used=_ngs_used_mask(options),
            ),
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
            pixel_scale_mas=float(result.metadata.pixel_scale_mas),
            tel_diameter_m=float(result.metadata.tel_diameter_m),
            tel_pupil=np.asarray(result.metadata.tel_pupil, dtype=np.float32),
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
        context.result.meta.update(dict(_require_hybrid_result(context).meta))

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
        return {self.EXTRA_STAT_JITTER: np.asarray(_require_hybrid_result(context).jitter_mas, dtype=np.float32)}

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
            "hybrid/mas2nm": {"dtype": "float32", "shape": ()},
            "hybrid/psd_valid_count": {"dtype": "int32", "shape": ()},
            "hybrid/psd_valid_fraction": {"dtype": "float32", "shape": ()},
            "hybrid/psd_valid_mask": {"dtype": "bool", "shape": ("num_sci",)},
            "hybrid/ctot_trace_mas2_min": {"dtype": "float32", "shape": ()},
            "hybrid/ctot_trace_mas2_max": {"dtype": "float32", "shape": ()},
            "hybrid/ctot_trace_mas2_mean": {"dtype": "float32", "shape": ()},
            "hybrid/ctot_det_mas4_min": {"dtype": "float32", "shape": ()},
            "hybrid/ctot_det_mas4_max": {"dtype": "float32", "shape": ()},
            "hybrid/ctot_det_mas4_mean": {"dtype": "float32", "shape": ()},
            "hybrid/ngs_used": {"dtype": "bool", "shape": ("num_ngs",)},
            "hybrid/ngs/ee": {"dtype": "float32", "shape": ("num_ngs",)},
            "hybrid/ngs/fwhm_mas": {"dtype": "float32", "shape": ("num_ngs",)},
            "hybrid/ngs/sr": {"dtype": "float32", "shape": ("num_ngs",)},
            "hybrid/ngs/flux_ph_s": {"dtype": "float32", "shape": ("num_ngs",)},
            "hybrid/ngs/frequency_hz": {"dtype": "float32", "shape": ("num_ngs",)},
        }
        if diagnostics_level == schema.DIAGNOSTICS_LEVEL_DEBUG:
            specs.update(
                {
                    "hybrid/ctot_nm2": {"dtype": "float32", "shape": ("num_sci", 2, 2)},
                    "hybrid/ctot_mas2": {"dtype": "float32", "shape": ("num_sci", 2, 2)},
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
        diagnostics_context = _require_hybrid_result(context).diagnostics_context
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
        """Build upstream Hybrid diagnostics for the configured level.

        ``validation`` diagnostics include compact scalar Ctot summaries, PSD
        validity summaries, slot-aligned active-NGS mask, and slot-aligned
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
        ctot_mas2 = np.asarray(diagnostics_context.ctot.ctot_mas2, dtype=float)
        trace = np.trace(ctot_mas2, axis1=-2, axis2=-1)
        det = np.linalg.det(ctot_mas2)
        used = np.asarray(diagnostics_context.ngs_used, dtype=bool)
        ngs_flux = _require_active_vector(diagnostics_context.ctot.ngs_flux, "ngs_flux")
        ngs_frequency = _require_active_vector(diagnostics_context.ctot.ngs_frequency, "ngs_frequency")
        diagnostics: dict[str, Any] = {
            "hybrid/mas2nm": np.float32(diagnostics_context.ctot.mas2nm),
            "hybrid/psd_valid_count": np.int32(np.count_nonzero(diagnostics_context.psd_mask)),
            "hybrid/psd_valid_fraction": np.float32(np.count_nonzero(diagnostics_context.psd_mask) / diagnostics_context.psd_mask.size),
            "hybrid/psd_valid_mask": np.asarray(diagnostics_context.psd_mask, dtype=bool),
            "hybrid/ctot_trace_mas2_min": np.float32(np.min(trace)),
            "hybrid/ctot_trace_mas2_max": np.float32(np.max(trace)),
            "hybrid/ctot_trace_mas2_mean": np.float32(np.mean(trace)),
            "hybrid/ctot_det_mas4_min": np.float32(np.min(det)),
            "hybrid/ctot_det_mas4_max": np.float32(np.max(det)),
            "hybrid/ctot_det_mas4_mean": np.float32(np.mean(det)),
            "hybrid/ngs_used": used,
            "hybrid/ngs/ee": _slot_aligned(diagnostics_context.metrics.ee, used),
            "hybrid/ngs/fwhm_mas": _slot_aligned(diagnostics_context.metrics.fwhm_mas, used),
            "hybrid/ngs/sr": _slot_aligned(diagnostics_context.metrics.sr, used),
            "hybrid/ngs/flux_ph_s": _slot_aligned(ngs_flux, used),
            "hybrid/ngs/frequency_hz": _slot_aligned(ngs_frequency, used),
        }
        if self.diagnostics_level == schema.DIAGNOSTICS_LEVEL_DEBUG:
            diagnostics.update(
                {
                    "hybrid/ctot_nm2": np.asarray(diagnostics_context.ctot.ctot_nm2, dtype=np.float32),
                    "hybrid/ctot_mas2": np.asarray(diagnostics_context.ctot.ctot_mas2, dtype=np.float32),
                    "hybrid/runtime_ini_text": str(diagnostics_context.ctot.runtime_ini_text or ""),
                }
            )
        return diagnostics

    # Provider hooks

    def _predict_science_psfs(self, setup: HybridSetup, options: Mapping[str, Any]) -> SciencePsfProviderResult:
        """Return science PSFs from the artifact-backed provider.

        Subclasses may override this hook to provide project-specific PSF
        sources or pixel-scale computation. Implementations must preserve their
        source PSF flux convention and return finite positive-flux PSFs.
        """
        x, y = polar_to_cartesian(setup.sci_r_arcsec, setup.sci_theta_deg)
        prediction = _evaluate_science_ho_psf_runtime_interpolator(
            self._get_science_ho_psf_runtime_interpolator(),
            zenith_angle_deg=_require_option_scalar(options, schema.KEY_OPTION_ZENITH_ANGLE_DEG),
            wavelength_um=_require_option_scalar(options, schema.KEY_OPTION_WAVELENGTH_UM),
            x_arcsec=x,
            y_arcsec=y,
        )
        _validate_psf_flux(prediction.psfs, label="science PSFs")
        return SciencePsfProviderResult(
            psfs=np.asarray(prediction.psfs, dtype=np.float32),
            metadata=prediction.metadata,
            meta=prediction.meta,
        )

    def _predict_ngs_metrics(self, active_ngs: _ActiveNgs, options: Mapping[str, Any]) -> NgsMetricProviderResult:
        """Return NGS-HO metrics from the artifact-backed provider."""
        prediction = evaluate_ngs_ho_metric_interpolator(
            self._get_ngs_ho_metric_interpolator(),
            zenith_angle_deg=_require_option_scalar(options, schema.KEY_OPTION_ZENITH_ANGLE_DEG),
            x_arcsec=active_ngs.x_arcsec,
            y_arcsec=active_ngs.y_arcsec,
        )
        return NgsMetricProviderResult(
            ee=np.asarray(prediction.ee, dtype=float),
            fwhm_mas=np.asarray(prediction.fwhm_mas, dtype=float),
            sr=np.asarray(prediction.sr, dtype=float),
        )

    # MASTSEL runtime

    def _compute_mastsel_ctot(
        self,
        parser: ConfigParser,
        setup: HybridSetup,
        active_ngs: _ActiveNgs,
        metrics: NgsMetricProviderResult,
    ) -> HybridCtotResult:
        """Call MASTSEL and return validated Ctot in ``nm^2`` and ``mas^2``."""
        MavisLO = _load_mavis_lo()
        with tempfile.TemporaryDirectory(prefix="ao_predict_hybrid_") as tmpdir:
            ini_path = Path(tmpdir) / "sim.ini"
            runtime_ini_text = _serialize_parser(parser)
            ini_path.write_text(runtime_ini_text, encoding="utf-8")
            mlo = MavisLO(str(ini_path.parent), ini_path.stem, verbose=False)
            if getattr(mlo, "error", False):
                raise RuntimeError("MASTSEL failed to load generated Hybrid runtime config.")
            science_coords = np.column_stack(polar_to_cartesian(setup.sci_r_arcsec, setup.sci_theta_deg))
            ngs_coords = np.column_stack([active_ngs.x_arcsec, active_ngs.y_arcsec])
            ngs_flux = self._ngs_flux_from_config(parser, active_ngs.mag, setup)
            ngs_frequency = np.full(active_ngs.mag.size, self._get_frame_rate_lo(parser), dtype=float)
            ctot_nm2 = np.asarray(
                mlo.computeTotalResidualMatrix(
                    science_coords,
                    ngs_coords,
                    ngs_flux,
                    ngs_frequency,
                    np.asarray(metrics.sr, dtype=float),
                    np.asarray(metrics.ee, dtype=float),
                    np.asarray(metrics.fwhm_mas, dtype=float),
                    aNGS_FWHM_DL_mas=None,
                    doAll=True,
                ),
                dtype=float,
            )
            validate_ctot_shape(ctot_nm2, expected_size=science_coords.shape[0], label="MASTSEL Ctot")
            mas2nm = getattr(mlo, "mas2nm", None)
            if mas2nm is None:
                raise ValueError("MASTSEL did not expose mas2nm conversion.")
            mas2nm = float(mas2nm)
            if not np.isfinite(mas2nm) or mas2nm <= 0.0:
                raise ValueError(f"MASTSEL returned invalid mas2nm conversion: {mas2nm!r}.")
            return HybridCtotResult(
                ctot_nm2=ctot_nm2,
                ctot_mas2=ctot_nm2 / mas2nm**2,
                mas2nm=mas2nm,
                ngs_flux=np.asarray(ngs_flux, dtype=float),
                ngs_frequency=np.asarray(ngs_frequency, dtype=float),
                runtime_ini_text=runtime_ini_text,
            )

    def _build_runtime_mastsel_parser(
        self,
        setup: HybridSetup,
        options: Mapping[str, Any],
        active_ngs: _ActiveNgs,
    ) -> ConfigParser:
        """Build a temporary MASTSEL INI parser for one option row."""
        parser = deepcopy(self.base_config.parser)
        self._update_atmosphere_in_ini(parser, options, setup)
        self._update_science_in_ini(parser, options, setup)
        self._update_ngs_in_ini(parser, active_ngs)
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
        the AO Predict runtime ``r0_m`` reaches MASTSEL directly.
        """
        if parser.has_section("telescope"):
            parser["telescope"]["ZenithAngle"] = f"{_require_option_scalar(options, schema.KEY_OPTION_ZENITH_ANGLE_DEG):.6g}"

        if not parser.has_section("atmosphere"):
            return
        atmosphere = parser["atmosphere"]
        profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
        profile = atm.select_atm_profile(setup.atm_profiles, profile_id)
        self._write_atmosphere_profile_fields(parser, profile)

        r0_m = _runtime_r0_m(setup, options)
        atmosphere["r0_Value"] = f"{r0_m:.6g}"
        if "Seeing" in atmosphere:
            del atmosphere["Seeing"]

    def _update_science_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: HybridSetup,
    ) -> None:
        """Apply science geometry and science-wavelength updates."""
        self._write_science_source_fields(
            parser,
            setup.sci_r_arcsec,
            setup.sci_theta_deg,
            _require_option_scalar(options, schema.KEY_OPTION_WAVELENGTH_UM),
        )

    def _update_ngs_in_ini(
        self,
        parser: ConfigParser,
        active_ngs: _ActiveNgs,
    ) -> None:
        """Apply active-NGS geometry and LO lenslet updates."""
        self._write_source_geometry_fields(parser, "sources_LO", active_ngs.r_arcsec, active_ngs.theta_deg)
        if parser.has_section("sensor_LO") and "NumberLenslets" in parser["sensor_LO"]:
            n_lenslets = self._get_n_lenslets_lo(parser)
            parser["sensor_LO"]["NumberLenslets"] = _format_ini_array(np.full(active_ngs.mag.size, n_lenslets, dtype=float))

    def _ngs_flux_from_config(self, parser: ConfigParser, ngs_mag: np.ndarray, setup: HybridSetup) -> np.ndarray:
        """Return active NGS flux in photons per second for MASTSEL."""
        photometry = self._get_ngs_photometry_config(parser, setup.ngs_mag_zeropoint)
        photons_per_frame = magnitudes_to_photons_per_frame(np.asarray(ngs_mag, dtype=float), photometry)
        return np.asarray(photons_per_frame, dtype=float) * float(photometry.frame_rate_hz)

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
        for key in (schema.KEY_OPTION_NGS_R_ARCSEC, schema.KEY_OPTION_NGS_THETA_DEG, schema.KEY_OPTION_NGS_MAG):
            if key not in options:
                raise ValueError(f"HybridSimulation options require '{key}'.")
        if schema.KEY_OPTION_NGS_USED not in options:
            raise ValueError(
                "Missing required runtime option 'ngs_used'. Call runner.prepare_options_payload(...) "
                "or api.init_dataset(...) so core derives runtime fields first."
            )
        r = as_float_vector(options[schema.KEY_OPTION_NGS_R_ARCSEC], label=schema.KEY_OPTION_NGS_R_ARCSEC)
        theta = as_float_vector(options[schema.KEY_OPTION_NGS_THETA_DEG], label=schema.KEY_OPTION_NGS_THETA_DEG)
        mag = as_float_vector(options[schema.KEY_OPTION_NGS_MAG], label=schema.KEY_OPTION_NGS_MAG)
        used = np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)
        if r.shape != theta.shape or r.shape != mag.shape or r.shape != used.shape:
            raise ValueError("HybridSimulation NGS option vectors and ngs_used must have identical shape.")
        if not np.any(used):
            raise ValueError("HybridSimulation requires at least one active NGS.")
        active_r = r[used]
        active_theta = theta[used]
        active_mag = mag[used]
        if not np.all(np.isfinite(active_r)) or not np.all(np.isfinite(active_theta)) or not np.all(np.isfinite(active_mag)):
            raise ValueError("HybridSimulation active NGS options must be finite.")
        x, y = polar_to_cartesian(active_r, active_theta)
        return _ActiveNgs(
            r_arcsec=active_r,
            theta_deg=active_theta,
            x_arcsec=x,
            y_arcsec=y,
            mag=active_mag,
        )


def apply_ctot_blur(
    psfs: np.ndarray,
    ctot_nm2: np.ndarray,
    *,
    pixel_scale_mas: float,
    mas2nm: float,
) -> None:
    """Apply finite-FOV Hybrid Ctot blur in place.

    Each PSF is zero-padded before applying the image-plane OTF, cropped back
    to the retained PSF field of view, and clipped for numerical negative
    pixels. The cropped result is not renormalized, so Ctot blur can move
    energy outside the retained PSF support.
    """
    validate_ctot_shape(ctot_nm2, expected_size=np.asarray(psfs).shape[0], label="Ctot")
    psfs_array = np.asarray(psfs)
    if psfs_array.ndim != 3:
        raise ValueError(f"psfs must have shape (N, y, x); got {psfs_array.shape}.")
    if not np.all(np.isfinite(psfs_array)):
        raise ValueError("psfs must contain only finite values.")
    if not np.isfinite(pixel_scale_mas) or float(pixel_scale_mas) <= 0.0:
        raise ValueError(f"pixel_scale_mas must be positive and finite, got {pixel_scale_mas!r}.")
    adapter = LowOrderMas2NmAdapter(mas2nm)
    input_flux = np.sum(psfs_array, axis=(-2, -1), dtype=np.float64)
    if not np.all(np.isfinite(input_flux)) or np.any(input_flux <= 0.0):
        raise ValueError("psfs must have strictly positive finite total flux.")

    diff_ctot = np.asarray(ctot_nm2, dtype=float)
    valid = psd_valid_mask(diff_ctot)
    if not np.any(valid):
        return

    covariance_pix2 = diff_ctot / adapter.mas2nm**2 / float(pixel_scale_mas) ** 2
    for index, covariance in enumerate(covariance_pix2):
        if not valid[index]:
            continue
        original_shape = tuple(psfs_array[index].shape)
        work_shape = tuple(2 * size for size in original_shape)
        padded = _centered_pad(psfs_array[index].astype(np.float32, copy=False), work_shape)
        otf = _image_plane_gaussian_otf(covariance, padded.shape)
        raw = np.real(np.fft.ifft2(np.fft.fft2(padded) * otf))
        raw = _center_crop(raw, original_shape)
        raw = np.clip(raw.astype(np.float32, copy=False), 0.0, None)
        raw_sum = float(np.sum(raw, dtype=np.float64))
        if raw_sum <= 0.0:
            raise ValueError(f"Ctot blur produced non-positive flux for PSF {index}.")
        psfs_array[index] = raw


def _centered_pad(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source_y, source_x = psf.shape
    target_y, target_x = shape
    if target_y < source_y or target_x < source_x:
        raise ValueError(f"Target padded shape {shape} cannot contain PSF shape {psf.shape}.")
    before_y = (target_y - source_y) // 2
    before_x = (target_x - source_x) // 2
    return np.pad(
        psf,
        (
            (before_y, target_y - source_y - before_y),
            (before_x, target_x - source_x - before_x),
        ),
        mode="constant",
    )


def _center_crop(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_y, target_x = shape
    source_y, source_x = psf.shape
    if target_y > source_y or target_x > source_x:
        raise ValueError(f"Target crop shape {shape} cannot exceed PSF shape {psf.shape}.")
    start_y = (source_y - target_y) // 2
    start_x = (source_x - target_x) // 2
    return psf[start_y : start_y + target_y, start_x : start_x + target_x]


def jitter_mas_from_ctot(ctot_mas2: np.ndarray) -> np.ndarray:
    """Return Hybrid jitter as ``sqrt(trace(ctot_mas2))``."""
    validate_ctot_shape(np.asarray(ctot_mas2, dtype=float), label="Ctot mas^2")
    traces = np.trace(np.asarray(ctot_mas2, dtype=float), axis1=-2, axis2=-1)
    return np.sqrt(np.clip(traces, 0.0, None))


def psd_valid_mask(ctot_nm2: np.ndarray) -> np.ndarray:
    """Return mask where Ctot is positive semidefinite and nonzero."""
    validate_ctot_shape(np.asarray(ctot_nm2, dtype=float), label="Ctot")
    sym = 0.5 * (ctot_nm2 + np.swapaxes(ctot_nm2, -1, -2))
    eig = np.linalg.eigvalsh(sym)
    return np.all(eig >= 0.0, axis=1) & np.any(eig > 1.0e-12, axis=1)


def validate_ctot_shape(ctot: np.ndarray, *, label: str, expected_size: int | None = None) -> None:
    """Validate a Ctot covariance cube."""
    if ctot.ndim != 3 or ctot.shape[-2:] != (2, 2):
        raise ValueError(f"{label} must have shape (N, 2, 2); got {ctot.shape}.")
    if expected_size is not None and ctot.shape[0] != int(expected_size):
        raise ValueError(f"{label} field length {ctot.shape[0]} does not match expected {expected_size}.")
    if not np.all(np.isfinite(ctot)):
        raise ValueError(f"{label} contains non-finite values.")


def polar_to_cartesian(r_arcsec: np.ndarray, theta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar field coordinates to Cartesian arcsecond coordinates."""
    r_arcsec = np.asarray(r_arcsec, dtype=float).reshape(-1)
    theta_deg = np.asarray(theta_deg, dtype=float).reshape(-1)
    if r_arcsec.shape != theta_deg.shape:
        raise ValueError(f"Coordinate shapes differ: {r_arcsec.shape} != {theta_deg.shape}.")
    theta_rad = np.deg2rad(theta_deg)
    return r_arcsec * np.cos(theta_rad), r_arcsec * np.sin(theta_rad)


def _require_hybrid_result(context: SimulationContext) -> _HybridRuntimeResult:
    result = context.runtime.get(HybridSimulation.KEY_RUNTIME_RESULT)
    if not isinstance(result, _HybridRuntimeResult):
        raise ValueError("Missing Hybrid runtime result. Did run(...) complete?")
    return result


def _science_meta_field_names(interpolator: ScienceHoPsfInterpolator) -> tuple[str, ...]:
    return tuple(str(name) for name in interpolator.meta.keys())


def _validate_science_meta_fields_match(
    simulation_payload: Mapping[str, Any],
    interpolator: ScienceHoPsfInterpolator,
) -> None:
    payload_fields = normalize_meta_field_names(simulation_payload.get(schema.KEY_SIMULATION_META_FIELDS, ()))
    artifact_fields = _science_meta_field_names(interpolator)
    if payload_fields != artifact_fields:
        raise ValueError(
            "HybridSimulation science meta field registry mismatch: "
            f"payload has {list(payload_fields)}, artifact has {list(artifact_fields)}."
        )


def _runtime_r0_m(setup: HybridSetup, options: Mapping[str, Any]) -> float:
    if schema.KEY_OPTION_R0_M in options:
        return _require_option_scalar(options, schema.KEY_OPTION_R0_M)
    profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
    profile = atm.select_atm_profile(setup.atm_profiles, profile_id)
    return float(profile[atm.KEY_SETUP_ATM_PROFILE_R0_M])


def _ngs_used_mask(options: Mapping[str, Any]) -> np.ndarray:
    """Return the runtime NGS slot mask."""
    if schema.KEY_OPTION_NGS_USED not in options:
        raise ValueError("Hybrid diagnostics require runtime option 'ngs_used'.")
    return np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)


def _require_active_vector(value: np.ndarray | None, label: str) -> np.ndarray:
    """Return a finite active-NGS diagnostic vector."""
    if value is None:
        raise ValueError(f"Hybrid diagnostics require {label}.")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Hybrid diagnostics {label} must contain finite values.")
    return arr


def _slot_aligned(active_values: np.ndarray, used: np.ndarray) -> np.ndarray:
    """Return NGS slot-aligned values with inactive slots set to ``NaN``."""
    active = np.asarray(active_values, dtype=float).reshape(-1)
    used = np.asarray(used, dtype=bool).reshape(-1)
    if active.shape[0] != int(np.count_nonzero(used)):
        raise ValueError("Hybrid diagnostics active NGS values do not match ngs_used.")
    out = np.full(used.shape, np.nan, dtype=np.float32)
    out[used] = active.astype(np.float32)
    return out


def _require_option_scalar(options: Mapping[str, Any], key: str) -> float:
    if key not in options:
        raise ValueError(f"HybridSimulation options require '{key}'.")
    value = float(np.asarray(options[key]).item())
    if not np.isfinite(value):
        raise ValueError(f"HybridSimulation option '{key}' must be finite.")
    return value


def _validate_psf_flux(psfs: np.ndarray, *, label: str) -> None:
    psfs = np.asarray(psfs)
    if psfs.ndim != 3:
        raise ValueError(f"{label} must have shape (N, y, x); got {psfs.shape}.")
    if not np.all(np.isfinite(psfs)):
        raise ValueError(f"{label} must contain only finite values.")
    flux = np.sum(psfs, axis=(-2, -1), dtype=np.float64)
    if not np.all(np.isfinite(flux)) or np.any(flux <= 0.0):
        raise ValueError(f"{label} must have strictly positive finite total flux.")


def _image_plane_gaussian_otf(covariance_pix2: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    covariance = 0.5 * (np.asarray(covariance_pix2, dtype=float) + np.asarray(covariance_pix2, dtype=float).T)
    exponent = -2.0 * np.pi**2 * (
        float(covariance[0, 0]) * fx_grid**2
        + float(covariance[1, 1]) * fy_grid**2
        + 2.0 * float(covariance[0, 1]) * fx_grid * fy_grid
    )
    return np.exp(exponent)


def _artifact_file_cache_key(path: Path) -> _ArtifactFileCacheKey:
    resolved_path = Path(path).expanduser().resolve()
    stat = resolved_path.stat()
    return _ArtifactFileCacheKey(
        path=str(resolved_path),
        size_bytes=int(stat.st_size),
        modified_ns=int(stat.st_mtime_ns),
    )


def _get_cached_science_ho_psf_runtime_interpolator(path: Path) -> _ScienceHoPsfRuntimeInterpolator:
    key = _artifact_file_cache_key(path)
    runtime_interpolator = _SCIENCE_HO_PSF_RUNTIME_INTERPOLATOR_CACHE.get(key)
    if runtime_interpolator is None:
        artifact = load_science_ho_psf_interpolator(path)
        validate_science_ho_psf_interpolator(artifact)
        runtime_interpolator = _prepare_science_ho_psf_runtime_interpolator(artifact)
        _SCIENCE_HO_PSF_RUNTIME_INTERPOLATOR_CACHE[key] = runtime_interpolator
    return runtime_interpolator


def _get_cached_ngs_ho_metric_interpolator(path: Path) -> NgsHoMetricInterpolator:
    key = _artifact_file_cache_key(path)
    interpolator = _NGS_HO_METRIC_INTERPOLATOR_CACHE.get(key)
    if interpolator is None:
        interpolator = load_ngs_ho_metric_interpolator(path)
        validate_ngs_ho_metric_interpolator(interpolator)
        _NGS_HO_METRIC_INTERPOLATOR_CACHE[key] = interpolator
    return interpolator


def _load_mavis_lo() -> Any:
    try:
        from mastsel import MavisLO  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - depends on optional runtime package.
        raise RuntimeError("MASTSEL is not importable; cannot run HybridSimulation.") from exc
    return MavisLO


__all__ = [
    "HybridCtotResult",
    "HybridDiagnosticsContext",
    "HybridSetup",
    "HybridSimulation",
    "LowOrderMas2NmAdapter",
    "NgsMetricProviderResult",
    "SciencePsfProviderResult",
    "apply_ctot_blur",
    "jitter_mas_from_ctot",
    "polar_to_cartesian",
    "psd_valid_mask",
    "validate_ctot_shape",
]
