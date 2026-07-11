"""TIPTOP simulation implementation."""

from __future__ import annotations

from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import atm
from . import schema
from .base import BaseSimulationSetup, PsfParameters
from .helpers import r0_to_seeing_arcsec
from .interfaces import SimulationContext, SimulationSetup
from .photometry import magnitudes_to_photons_per_frame
from .tiptop_config_backed import (
    TiptopBaseConfig,
    TiptopConfigBackedSimulation,
    _format_ini_array,
    _serialize_parser,
)
from ..utils import as_float_vector


@dataclass(frozen=True)
class TiptopSetup(BaseSimulationSetup):
    """Typed setup payload for ``TiptopSimulation``.

    This currently adds no fields beyond ``BaseSimulationSetup`` but keeps a
    dedicated type for TIPTOP-specific extensions.
    """


# Tiptop Simulation

class TiptopSimulation(TiptopConfigBackedSimulation):
    """TIPTOP-backed `Simulation` implementation."""

    _VERSION = "0.0.1"

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
        if schema.KEY_OPTION_ZENITH_ANGLE_DEG in options and parser.has_section("telescope"):
            parser["telescope"]["ZenithAngle"] = f"{float(options[schema.KEY_OPTION_ZENITH_ANGLE_DEG]):.6g}"

        if not parser.has_section("atmosphere"):
            return

        atmosphere_section = parser["atmosphere"]

        if schema.KEY_OPTION_ATM_PROFILE_ID in options:
            atm_profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
            atm_profile = atm.select_atm_profile(setup.atm_profiles, atm_profile_id)
            self._write_atmosphere_profile_fields(parser, atm_profile)

        if schema.KEY_OPTION_R0_M in options:
            r0_m = float(options[schema.KEY_OPTION_R0_M])
            atm_wavelength_m = self._get_required_atm_wavelength_m(parser, "convert r0_m to Seeing")
            atmosphere_section["Seeing"] = f"{r0_to_seeing_arcsec(r0_m, atm_wavelength_m):.6g}"
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
        wavelength_um = float(options[schema.KEY_OPTION_WAVELENGTH_UM]) if schema.KEY_OPTION_WAVELENGTH_UM in options else None
        self._write_science_source_fields(parser, setup.sci_r_arcsec, setup.sci_theta_deg, wavelength_um)

    def _update_lgs_in_ini(self, parser: ConfigParser, setup: TiptopSetup) -> None:
        """Apply invariant LGS geometry from setup."""
        if setup.lgs_r_arcsec.size > 0 and parser.has_section("sources_HO"):
            self._write_source_geometry_fields(parser, "sources_HO", setup.lgs_r_arcsec, setup.lgs_theta_deg)

    def _update_ngs_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply active-NGS geometry and photon updates."""
        required_ngs_keys = (
            schema.KEY_OPTION_NGS_R_ARCSEC,
            schema.KEY_OPTION_NGS_THETA_DEG,
            schema.KEY_OPTION_NGS_MAG,
        )
        if not all(key in options for key in required_ngs_keys):
            return
        if schema.KEY_OPTION_NGS_USED not in options:
            raise ValueError(
                "Missing required runtime option 'ngs_used' while applying NGS per-simulation overrides. "
                "Call runner.prepare_options_payload(...) (or api.init_dataset(...)) so core derives runtime fields first."
            )

        ngs_mag = np.asarray(options[schema.KEY_OPTION_NGS_MAG], dtype=float).reshape(-1)
        ngs_used = np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)
        if not np.any(ngs_used):
            return
        photometry = self._get_ngs_photometry_config(parser, setup.ngs_mag_zeropoint)

        if parser.has_section("sources_LO"):
            self._write_source_geometry_fields(
                parser,
                "sources_LO",
                as_float_vector(options[schema.KEY_OPTION_NGS_R_ARCSEC], label=schema.KEY_OPTION_NGS_R_ARCSEC)[ngs_used],
                as_float_vector(options[schema.KEY_OPTION_NGS_THETA_DEG], label=schema.KEY_OPTION_NGS_THETA_DEG)[ngs_used],
            )

        photons_per_frame = magnitudes_to_photons_per_frame(
            ngs_mag[ngs_used],
            photometry,
        )
        if parser.has_section("sensor_LO"):
            parser["sensor_LO"]["NumberPhotons"] = _format_ini_array(np.round(photons_per_frame, 0))
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
        ee_apertures_mas = np.asarray(setup.ee_apertures_mas, dtype=float).reshape(-1)
        if ee_apertures_mas.size == 0:
            raise ValueError("ee_apertures_mas must contain at least one value.")
        ee_radius_mas = float(ee_apertures_mas[0]) * 0.5
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
            PSF cube with shape ``[M, Ny, Nx]`` or ``None`` when TIPTOP did
            not expose per-science sampling results.

        Raises:
            ValueError: If the TIPTOP runtime object is missing.
        """
        simulation = context.runtime.get(self.KEY_RUNTIME_SIMULATION)
        if simulation is None:
            raise ValueError("Missing TIPTOP simulation in context.runtime. Did you call run()?")

        psfs: np.ndarray | None = None
        if hasattr(simulation, "results"):
            results = getattr(simulation, "results")
            psfs = np.asarray(
                [
                    np.asarray(getattr(item, "sampling", item), dtype=np.float32)
                    for item in results
                ],
                dtype=np.float32,
            )

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
            pixel_scale_mas = float(getattr(simulation, "psInMas"))
        elif parser.has_section("sensor_science") and "PixelScale" in parser["sensor_science"]:
            pixel_scale_mas = float(parser["sensor_science"]["PixelScale"])
        else:
            raise ValueError("Unable to resolve pixel scale (mas) from TIPTOP outputs or INI.")

        if hasattr(simulation, "tel_radius"):
            tel_diameter_m = float(getattr(simulation, "tel_radius")) * 2.0
        elif parser.has_section("telescope") and "TelescopeDiameter" in parser["telescope"]:
            tel_diameter_m = float(parser["telescope"]["TelescopeDiameter"])
        else:
            raise ValueError("Unable to resolve telescope diameter (m) from TIPTOP outputs or INI.")

        try:
            tel_pupil = np.asarray(simulation.fao.ao.tel.pupil, dtype=np.float32)
        except Exception as exc:
            raise ValueError("Unable to resolve telescope pupil from TIPTOP output object.") from exc

        return PsfParameters(
            pixel_scale_mas=pixel_scale_mas,
            tel_diameter_m=tel_diameter_m,
            tel_pupil=tel_pupil,
        )
