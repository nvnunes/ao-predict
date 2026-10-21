"""TIPTOP simulation implementation."""

from __future__ import annotations

from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from astropy import units as u
from ngs_photometry import magnitudes_to_photons_per_frame

from . import atm
from . import schema
from .base import BaseSimulationSetup, PsfParameters
from .helpers import r0_to_seeing
from .interfaces import SimulationContext, SimulationSetup
from .tiptop_config_backed import (
    TiptopConfigBackedSimulation,
    _format_ini_array,
    _serialize_parser,
)
from .._units import quantity_value


@dataclass(frozen=True)
class TiptopSetup(BaseSimulationSetup):
    """Typed setup payload for ``TiptopSimulation``.

    This currently adds no fields beyond ``BaseSimulationSetup`` but keeps a
    dedicated type for TIPTOP-specific extensions.
    """


def tiptop_full_flux_denominators(simulation: Any, *, skip_reshape: bool = False) -> np.ndarray:
    """Return conserved pre-crop TIPTOP power in detector-pixel units.

    Args:
        simulation: Completed TIPTOP runtime exposing ``N``, the resampled
            ``mask.sampling``, ``wvl``, ``nWvl``, ``wvlMax``, ``overSamp`` and
            ``opdMap``. Values follow TIPTOP's native, unit-free conventions.
        skip_reshape: Whether propagation skipped detector-pixel averaging.
            Science propagation uses the default ``False``.

    Returns:
        Float64 vector of length ``nWvl`` in wavelength order. Divide raw
        PSF pixels by the corresponding value before any additional crop.
        This is power on the complete sampled propagation plane, not an
        estimate from an internally cropped output image.

    Raises:
        ValueError: If retained propagation state is missing or invalid.

    MASTSEL uses an unscaled forward FFT for OTF-to-PSF propagation, giving
    ``N_prop**2 * OTF(0)`` total power. The turbulence OTF has unit DC. Without
    an OPD map, the telescope OTF's DC is pupil power divided by ``N_prop**2``.
    The OPD branch instead divides its telescope OTF by its maximum, giving
    unit DC. Detector rebinning takes means, so divides power by the square
    of the effective oversampling. Padding and subpixel-shift boundary losses
    never change this denominator.
    """
    try:
        n = int(simulation.N)
        mask = _tiptop_cpu_array(simulation.mask.sampling)
        wavelengths = np.atleast_1d(np.asarray(simulation.wvl, dtype=np.float64))
        oversampling = np.broadcast_to(
            np.asarray(simulation.overSamp, dtype=np.float64), wavelengths.shape,
        ).copy()
        n_wavelengths = int(simulation.nWvl)
        opd_map = simulation.opdMap
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("TIPTOP is missing valid retained propagation state.") from exc
    if n <= 0 or n % 2 or mask.shape != (n, n) or not np.all(np.isfinite(mask)):
        raise ValueError("TIPTOP propagation requires a finite pupil on an even N by N grid.")
    pupil_power = np.sum(np.square(np.abs(mask), dtype=np.float64), dtype=np.float64)
    if not np.isfinite(pupil_power) or pupil_power <= 0.0:
        raise ValueError("TIPTOP propagated pupil power must be positive and finite.")
    if (wavelengths.ndim != 1 or wavelengths.size != n_wavelengths
            or not np.all(np.isfinite(wavelengths) & (wavelengths > 0.0))):
        raise ValueError("TIPTOP wvl must contain nWvl positive finite wavelengths.")
    if not np.all(np.isfinite(oversampling) & (oversampling >= 1.0)
                  & (oversampling == np.floor(oversampling))):
        raise ValueError("TIPTOP oversampling must contain positive integers.")

    propagation_sizes = np.full(wavelengths.shape, n, dtype=np.float64)
    if n_wavelengths > 1:
        try:
            wavelength_ref = float(simulation.wvlMax)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("TIPTOP multi-wavelength propagation requires wvlMax.") from exc
        if not np.isfinite(wavelength_ref) or wavelength_ref < np.max(wavelengths):
            raise ValueError("TIPTOP wvlMax must be finite and at least max(wvl).")
        ratios = wavelength_ref / wavelengths
        factors = np.ceil(ratios)
        # Match psdSetToPsfSet's padPSD branch, including its even-grid rounding.
        propagation_sizes = np.round(n * (factors / ratios) / 2.0) * 2.0
        oversampling *= factors

    denominators = propagation_sizes**2 if opd_map is not None else np.full(
        wavelengths.shape, pupil_power, dtype=np.float64,
    )
    if not skip_reshape:
        denominators /= oversampling**2
    if not np.all(np.isfinite(denominators) & (denominators > 0.0)):
        raise ValueError("TIPTOP full-flux denominators must be positive and finite.")
    return denominators


def _tiptop_cpu_array(value: Any) -> np.ndarray:
    """Read NumPy or CuPy runtime values without changing their precision."""
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _validate_tiptop_psfs(psfs: np.ndarray) -> None:
    """Require finite, nonnegative crops on the conserved unit-flux scale."""
    if (psfs.ndim != 3 or psfs.size == 0
            or not np.all(np.isfinite(psfs)) or np.any(psfs < 0.0)):
        raise ValueError("TIPTOP PSFs must be a finite nonnegative [M, Ny, Nx] cube.")
    flux = np.sum(psfs, axis=(-2, -1), dtype=np.float64)
    if np.any(flux <= 0.0) or np.any(flux > 1.0 + 1.0e-5):
        raise ValueError("TIPTOP PSF in-frame flux must be positive and <= 1 on the conserved scale.")


# Tiptop Simulation

class TiptopSimulation(TiptopConfigBackedSimulation):
    """TIPTOP-backed `Simulation` implementation."""

    _VERSION = "0.0.2"

    KEY_RUNTIME_EFFECTIVE_PARSER = "effective_parser"
    KEY_RUNTIME_SIMULATION = "tiptop_simulation"

    # Setup payload lifecycle

    def _config_file_description(self) -> str:
        """Return the user-facing description for missing TIPTOP INI errors."""
        return "TIPTOP INI file"

    def _parse_setup_payload(self, setup_payload: Mapping[str, Any]) -> TiptopSetup:
        """Parse and validate TIPTOP persisted ``/setup`` without binding it.

        Args:
            setup_payload: Candidate persisted setup payload.

        Returns:
            Parsed ``TiptopSetup`` instance ready to be bound by
            ``load_setup_payload()``.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        return self._parse_base_setup_payload(setup_payload, TiptopSetup)

    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Bind typed TIPTOP setup from persisted ``/setup`` payload.

        Args:
            setup_payload: Persisted setup payload read from dataset storage.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        self._load_base_setup_payload(setup_payload, TiptopSetup)

    # Runtime INI update helpers

    def _create_runtime_context(self, index: int, options: dict[str, Any], setup: SimulationSetup) -> dict[str, Any]:
        """Create runtime scratch state for one TIPTOP simulation.

        This derives the per-simulation effective INI parser by copying the
        loaded base config and applying setup- and option-dependent runtime
        overrides.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Bound typed setup object for this simulation instance.

        Returns:
            Runtime scratch mapping containing the effective TIPTOP parser.

        Raises:
            TypeError: If ``setup`` is not ``TiptopSetup``.
            ValueError: If runtime INI updates require missing or invalid
                inputs.
        """
        del index
        if not isinstance(setup, TiptopSetup):
            raise TypeError("TiptopSimulation setup must be TiptopSetup.")
        # Per-simulation INI edits must not mutate the shared base parser.
        parser = deepcopy(self.base_config.parser)
        self._update_atmosphere_in_ini(parser, options, setup)
        self._update_science_in_ini(parser, options, setup)
        self._update_lgs_in_ini(parser, setup)
        self._update_ngs_in_ini(parser, options, setup)
        return {self.KEY_RUNTIME_EFFECTIVE_PARSER: parser}

    def _update_atmosphere_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply atmosphere/profile/r0 runtime updates."""
        if schema.KEY_OPTION_ZENITH_ANGLE in options and parser.has_section("telescope"):
            parser["telescope"]["ZenithAngle"] = f"{options[schema.KEY_OPTION_ZENITH_ANGLE].to_value(u.deg):.6g}"

        if not parser.has_section("atmosphere"):
            return

        atmosphere_section = parser["atmosphere"]

        if schema.KEY_OPTION_ATM_PROFILE_ID in options:
            atm_profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
            atm_profile = atm.select_atm_profile(setup.atm_profiles, atm_profile_id)
            self._write_atmosphere_profile_fields(parser, atm_profile)

        if schema.KEY_OPTION_R0 in options:
            r0 = float(options[schema.KEY_OPTION_R0].to_value(u.m))
            atm_wavelength_m = self._get_required_atm_wavelength_m(parser, "convert r0 to Seeing")
            atmosphere_section["Seeing"] = f"{r0_to_seeing(r0 * u.m, atm_wavelength_m * u.m).to_value(u.arcsec):.6g}"
            # TIPTOP runtime input is written as Seeing rather than r0_Value.
            if "r0_Value" in atmosphere_section:
                del atmosphere_section["r0_Value"]

    def _update_science_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply science geometry and science-wavelength updates."""
        wavelength = options.get(schema.KEY_OPTION_WAVELENGTH)
        self._write_science_source_fields(parser, setup.sci_r, setup.sci_theta, wavelength)

    def _update_lgs_in_ini(self, parser: ConfigParser, setup: TiptopSetup) -> None:
        """Apply invariant LGS geometry from setup."""
        if setup.lgs_r.size > 0 and parser.has_section("sources_HO"):
            self._write_source_geometry_fields(parser, "sources_HO", setup.lgs_r, setup.lgs_theta)

    def _update_ngs_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply active-NGS geometry and photon updates."""
        required_ngs_keys = (
            schema.KEY_OPTION_NGS_R,
            schema.KEY_OPTION_NGS_THETA,
            schema.KEY_OPTION_NGS_MAGNITUDE,
        )
        if not all(key in options for key in required_ngs_keys):
            return
        if schema.KEY_OPTION_NGS_USED not in options:
            raise ValueError(
                "Missing required runtime option 'ngs_used' while applying NGS per-simulation overrides. "
                "Call runner.prepare_options_payload(...) (or api.init_dataset(...)) so core derives runtime fields first."
            )

        ngs_magnitude = quantity_value(options[schema.KEY_OPTION_NGS_MAGNITUDE], u.mag, label="options.ngs_magnitude", dtype=float).reshape(-1)
        ngs_used = np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)
        if not np.any(ngs_used):
            return
        photometry = self._get_ngs_photometry_config(parser, setup.ngs_magnitude_zeropoint)

        if parser.has_section("sources_LO"):
            self._write_source_geometry_fields(
                parser,
                "sources_LO",
                quantity_value(options[schema.KEY_OPTION_NGS_R], u.arcsec, label=schema.KEY_OPTION_NGS_R)[ngs_used] * u.arcsec,
                quantity_value(options[schema.KEY_OPTION_NGS_THETA], u.deg, label=schema.KEY_OPTION_NGS_THETA)[ngs_used] * u.deg,
            )

        photons_per_frame = magnitudes_to_photons_per_frame(
            ngs_magnitude[ngs_used] * u.mag,
            photometry,
        )
        if parser.has_section("sensor_LO"):
            parser["sensor_LO"]["NumberPhotons"] = _format_ini_array(
                np.round(photons_per_frame.to_value(u.photon), 0)
            )
            if "NumberLenslets" in parser["sensor_LO"]:
                parser["sensor_LO"]["NumberLenslets"] = _format_ini_array(
                    np.full((int(np.sum(ngs_used)),), photometry.n_channels, dtype=float)
                )

    # Runtime lifecycle

    def _run_tiptop(self, simulation: Any) -> None:
        """Execute one constructed TIPTOP simulation.

        Subclasses may override this hook to select supported TIPTOP execution
        options without duplicating the surrounding temporary-INI lifecycle.

        Args:
            simulation: Constructed ``tiptop.baseSimulation`` instance. The
                hook must execute it in place and return only after the
                runtime results needed by extraction hooks are available.
        """
        simulation.doOverallSimulation()

    def run(self, context: SimulationContext) -> None:
        """Execute TIPTOP simulation and cache raw output object in runtime.

        Args:
            context: Simulation context produced by ``create()``.
        """
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")

        setup = context.setup
        if not isinstance(setup, TiptopSetup):
            raise TypeError("context.setup must be TiptopSetup.")
        ee_apertures = quantity_value(
            setup.ee_apertures,
            u.mas,
            label="setup.ee_apertures",
            dtype=float,
        ).reshape(-1)
        if ee_apertures.size == 0:
            raise ValueError("ee_apertures must contain at least one value.")
        ee_radius_mas = float(ee_apertures[0]) * 0.5
        ini_text = _serialize_parser(parser)

        with tempfile.TemporaryDirectory(prefix="ao_predict_tiptop_") as tmpdir:
            ini_path = Path(tmpdir) / "sim.ini"
            ini_path.write_text(ini_text, encoding="utf-8")

            path2param = str(ini_path.parent)
            parameters_file = ini_path.stem
            output_dir = str(ini_path.parent)
            output_file = ini_path.stem

            def _cpu_array_safe(v: Any) -> Any:
                return v.get() if hasattr(v, "get") else v

            # Compatibility patch for upstream astro-tiptop CPU-only runs where
            # cpuArray may receive numpy values without a `.get()` method.
            try:  # pragma: no cover - depends on external package internals
                import tiptop.tiptopUtils as tiptop_utils  # pylint: disable=import-outside-toplevel

                tiptop_utils.cpuArray = _cpu_array_safe
            except Exception:
                pass
            try:  # pragma: no cover - depends on external package internals
                import tiptop.baseSimulation as tiptop_base_module  # pylint: disable=import-outside-toplevel

                tiptop_base_module.cpuArray = _cpu_array_safe
            except Exception:
                pass

            from tiptop.tiptop import baseSimulation  # pylint: disable=import-outside-toplevel

            simulation = baseSimulation(
                path2param,
                parameters_file,
                output_dir,
                output_file,
                doConvolve=True,
                getHoErrorBreakDown=True,
                ensquaredEnergy=True,
                eeRadiusInMas=ee_radius_mas,
                doPlot=False,
                verbose=False,
            )
            self._run_tiptop(simulation)
            context.runtime[self.KEY_RUNTIME_SIMULATION] = simulation

    def _extract_psfs(self, context: SimulationContext) -> np.ndarray | None:
        """Extract the PSF cube from the completed TIPTOP runtime object.

        Args:
            context: Completed simulation context.

        Returns:
            Conserved-flux PSF cube with shape ``[M, Ny, Nx]``.

        Raises:
            ValueError: If runtime outputs or conserved-flux state are invalid.
        """
        simulation = context.runtime.get(self.KEY_RUNTIME_SIMULATION)
        if simulation is None:
            raise ValueError("Missing TIPTOP simulation in context.runtime. Did you call run()?")

        return self._extract_tiptop_psfs(simulation)

    def _extract_tiptop_psfs(self, simulation: Any) -> np.ndarray:
        """Extract one wavelength's final PSFs on the conserved flux scale.

        Args:
            simulation: Completed stock TIPTOP runtime. Its known convolution
                normalization is reversed using retained pre-convolution HO
                image power, never final-image power.

        Returns:
            Independent ``[M, Ny, Nx]`` cube in this simulation's precision.
            Small interpolation negatives are clipped before persistence,
            without renormalizing the crop.

        Raises:
            ValueError: If raw state is unavailable, multiple wavelengths were
                executed, or pixels violate the conserved-flux contract.
        """
        try:
            if simulation.nWvl != 1:
                raise ValueError("AO Predict PSF cubes require one TIPTOP science wavelength per simulation.")
            convolved_lo = simulation.LOisOn and simulation.doConvolve
            if convolved_lo and not simulation.doConvolveAsterism:
                raise ValueError("TIPTOP deferred asterism convolution has no final PSFs.")
            psfs = np.asarray([
                _tiptop_cpu_array(image.sampling) for image in simulation.results
            ], dtype=np.float64)
            # Stock convolve divides its input by its crop sum. Its Gaussian
            # transfer kernels have unit DC, including a second jitter pass.
            restore_scale = (
                convolved_lo or (not simulation.LOisOn and simulation.jitter_FWHM is not None)
            )
            if restore_scale:
                ho_psfs = np.asarray([
                    _tiptop_cpu_array(image.sampling) for image in simulation.psfLongExpPointingsArr
                ], dtype=np.float64)
                if ho_psfs.shape != psfs.shape or not np.all(np.isfinite(ho_psfs)):
                    raise ValueError("TIPTOP retained HO PSFs do not match final PSFs.")
                ho_power = np.sum(ho_psfs, axis=(-2, -1), dtype=np.float64)
                if np.any(ho_power <= 0.0):
                    raise ValueError("TIPTOP retained HO PSFs must have positive power.")
                psfs *= ho_power[:, None, None]
        except AttributeError as exc:
            raise ValueError("TIPTOP is missing final PSFs or retained pre-convolution state.") from exc
        return self._normalize_tiptop_psfs(psfs, simulation)

    def _normalize_tiptop_psfs(self, psfs: np.ndarray, simulation: Any) -> np.ndarray:
        """Normalize raw TIPTOP crops before persistence or further cropping.

        Args:
            psfs: Unnormalized ``[M, Ny, Nx]`` science PSFs, before TIPTOP's
                convolution input division or with that scaling restored.
            simulation: Completed runtime used by
                :func:`tiptop_full_flux_denominators`.

        Returns:
            Independent cube in this simulation's precision. Interpolation
            negatives up to 1e-6 per pixel and 1e-5 integrated full-flux
            fraction are clipped; the remaining crop is never sum-normalized.

        Raises:
            ValueError: If wavelengths, negative power or crop flux violate
                the conserved-flux contract.
        """
        denominators = tiptop_full_flux_denominators(simulation)
        if denominators.size != 1:
            raise ValueError("AO Predict PSF cubes require one TIPTOP science wavelength per simulation.")
        psfs = np.array(psfs, dtype=np.float64, copy=True)
        psfs /= denominators[0]
        if psfs.ndim != 3 or not np.all(np.isfinite(psfs)):
            raise ValueError("TIPTOP PSFs must be a finite [M, Ny, Nx] cube.")
        # Cubic subpixel shifts can introduce negatives beyond roundoff. Bound
        # both their individual amplitude and integrated full-flux fraction.
        negative = np.minimum(psfs, 0.0)
        if np.any(negative < -1.0e-6) or np.any(-negative.sum(axis=(-2, -1)) > 1.0e-5):
            raise ValueError("TIPTOP PSFs contain significant negative interpolation power.")
        psfs = np.maximum(psfs, 0.0).astype(self.dtype)
        _validate_tiptop_psfs(psfs)
        return psfs

    def prepare_psfs_for_stats(
        self,
        psfs: np.ndarray,
        setup: SimulationSetup | Mapping[str, Any],
        meta: Mapping[str, Any],
    ) -> np.ndarray:
        """Validate conserved-flux TIPTOP crops without rescaling their pixels.

        PSFs must already be finite, nonnegative, and have positive in-frame
        flux no greater than one (within numerical tolerance). The setup and
        metadata do not alter this normalization. Returns the original cube.
        """
        del setup, meta
        _validate_tiptop_psfs(np.asarray(psfs))
        return psfs

    def _extract_psf_parameters(self, context: SimulationContext) -> PsfParameters:
        """Extract persisted PSF metadata from completed TIPTOP runtime state.

        Args:
            context: Completed simulation context.

        Returns:
            Pixel scale, telescope diameter, and telescope pupil associated
            with the extracted PSFs.

        Raises:
            TypeError: If the effective parser is missing or invalid.
            ValueError: If required TIPTOP runtime outputs are unavailable.
        """
        simulation = context.runtime.get(self.KEY_RUNTIME_SIMULATION)
        if simulation is None:
            raise ValueError("Missing TIPTOP simulation in context.runtime. Did you call run()?")
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")

        if hasattr(simulation, "psInMas"):
            pixel_scale = float(getattr(simulation, "psInMas"))
        elif parser.has_section("sensor_science") and "PixelScale" in parser["sensor_science"]:
            pixel_scale = float(parser["sensor_science"]["PixelScale"])
        else:
            raise ValueError("Unable to resolve pixel scale (mas) from TIPTOP outputs or INI.")

        if hasattr(simulation, "tel_radius"):
            tel_diameter = float(getattr(simulation, "tel_radius")) * 2.0
        elif parser.has_section("telescope") and "TelescopeDiameter" in parser["telescope"]:
            tel_diameter = float(parser["telescope"]["TelescopeDiameter"])
        else:
            raise ValueError("Unable to resolve telescope diameter (m) from TIPTOP outputs or INI.")

        try:
            tel_pupil = np.asarray(simulation.fao.ao.tel.pupil, dtype=self.dtype)
        except Exception as exc:
            raise ValueError("Unable to resolve telescope pupil from TIPTOP output object.") from exc

        return PsfParameters(
            pixel_scale=pixel_scale * u.mas,
            tel_diameter=tel_diameter * u.m,
            tel_pupil=tel_pupil * u.dimensionless_unscaled,
        )
