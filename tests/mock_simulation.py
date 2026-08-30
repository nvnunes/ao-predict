"""Deterministic mock simulation for pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy import units as u

from ao_predict.simulation import schema
from ao_predict.simulation import atm
from ao_predict.simulation.base import BaseSimulation, BaseSimulationSetup, PsfParameters
from ao_predict.simulation.interfaces import SimulationContext, SimulationSetup


# Data structures

@dataclass(frozen=True)
class MockSetup(BaseSimulationSetup):
    """Typed setup payload used by ``MockSimulation`` tests."""


# Simulation implementation

class MockSimulation(BaseSimulation):
    """Deterministic simulation backend used for integration tests.

    This backend exercises the full ao-predict simulation lifecycle without
    depending on an external simulator.
    """

    _VERSION = "1.0"

    # Simulation payload lifecycle

    def prepare_simulation_payload(
        self,
        base_simulation_payload: Mapping[str, Any],
        simulation_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build persisted ``/simulation`` payload for the mock backend."""
        return self._build_simulation_payload(base_simulation_payload, simulation_cfg)

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Accept any persisted mock simulation payload.

        The mock backend has no simulation-specific persisted fields beyond the
        core ``/simulation`` contract.
        """
        super().validate_simulation_payload(simulation_payload)

    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Load persisted simulation payload for the mock backend."""
        del simulation_payload

    # Setup payload lifecycle

    def _default_atm_profile(self) -> dict[str, Any]:
        """Return deterministic fallback atmosphere profile for mock tests."""
        return {
            atm.KEY_SETUP_ATM_PROFILE_NAME: "default",
            atm.KEY_SETUP_ATM_PROFILE_R0: 0.16 * u.m,
            atm.KEY_SETUP_ATM_PROFILE_L0: 25.0 * u.m,
            atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS: np.asarray([0.0, 5000.0], dtype=float) * u.m,
            atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: np.asarray([0.6, 0.4], dtype=float) * u.dimensionless_unscaled,
            atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED: np.asarray([5.0, 10.0], dtype=float) * u.m / u.s,
            atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION: np.asarray([0.0, 90.0], dtype=float) * u.deg,
        }

    def prepare_setup_payload(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Prepare persisted ``/setup`` payload with deterministic defaults.

        The mock backend supplies stable defaults so tests can focus on core
        lifecycle behavior instead of simulator-specific configuration.
        """
        return self._build_setup_payload(
            base_setup_payload,
            setup_cfg,
            default_atm_wavelength=0.5 * u.um,
            default_atm_profile=self._default_atm_profile(),
            default_lgs_r=[] * u.arcsec,
            default_lgs_theta=[] * u.deg,
            default_sci_r=[0.0] * u.arcsec,
            default_sci_theta=[0.0] * u.deg,
            default_ngs_mag_zeropoint=1.0e10 * u.photon / u.s,
        )

    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Bind typed mock setup from persisted ``/setup`` payload.

        Args:
            setup_payload: Persisted setup payload read from dataset storage.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        self._load_base_setup_payload(setup_payload, MockSetup)

    def _parse_setup_payload(self, setup_payload: Mapping[str, Any]) -> MockSetup:
        """Parse and validate mock persisted ``/setup`` without binding it.

        Args:
            setup_payload: Candidate persisted setup payload.

        Returns:
            Parsed ``MockSetup`` instance ready to be bound by
            ``load_setup_payload()``.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        return self._parse_base_setup_payload(setup_payload, MockSetup)

    # Options payload lifecycle

    def prepare_options_payload(
        self,
        num_sims: int,
        setup_payload: Mapping[str, Any],
        base_options_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Prepare persisted ``/options`` payload with deterministic defaults."""
        del setup_payload
        default_options = {
            schema.KEY_OPTION_WAVELENGTH: 1.65 * u.um,
            schema.KEY_OPTION_ZENITH_ANGLE: 20.0 * u.deg,
            schema.KEY_OPTION_ATM_PROFILE_ID: np.int32(0),
            schema.KEY_OPTION_R0: 0.16 * u.m,
        }
        options_payload = {str(key): value.copy() for key, value in base_options_payload.items()}
        if not any(key in options_payload for key in schema.OPTION_KEYS_NGS):
            options_payload[schema.KEY_OPTION_NGS_R] = np.full((int(num_sims), 1), 1.0, dtype=float) * u.arcsec
            options_payload[schema.KEY_OPTION_NGS_THETA] = np.full((int(num_sims), 1), 0.0, dtype=float) * u.deg
            options_payload[schema.KEY_OPTION_NGS_MAGNITUDE] = np.full((int(num_sims), 1), 15.0, dtype=float) * u.mag
        return self._build_options_payload(
            int(num_sims),
            options_payload,
            default_options=default_options,
        )

    def _create_runtime_context(self, index: int, options: dict[str, Any], setup: SimulationSetup) -> dict[str, Any]:
        """Create deterministic runtime scratch state for one simulation.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Bound setup object for this simulation instance.

        Returns:
            Runtime scratch mapping consumed by the mock ``run()`` path.
        """
        del index, options, setup
        return {}

    def run(self, context: SimulationContext) -> None:
        """No-op runtime execution for the deterministic mock backend."""
        del context

    def _extract_psfs(self, context: SimulationContext) -> np.ndarray | None:
        """Return deterministic PSFs derived from the simulation index.

        Args:
            context: Completed simulation context.

        Returns:
            Deterministic PSF cube with shape ``[M, 4, 4]``.
        """
        num_sci = int(np.asarray(context.setup.sci_r, dtype=float).reshape(-1).shape[0])
        base = float(context.index + 1)
        return np.full((num_sci, 4, 4), 0.1 * base, dtype=np.float32)

    def _extract_psf_parameters(self, context: SimulationContext) -> PsfParameters:
        """Return deterministic PSF metadata for mock outputs.

        Args:
            context: Completed simulation context.

        Returns:
            Fixed pixel scale, telescope diameter, and telescope pupil values
            used by mock integration tests.
        """
        del context
        return PsfParameters(
            pixel_scale=5.0 * u.mas,
            tel_diameter=8.0 * u.m,
            tel_pupil=np.ones((4, 4), dtype=np.float32) * u.dimensionless_unscaled,
        )


class WarmupMockSimulation(MockSimulation):
    """Mock backend that requires worker warmup before execution."""

    def __init__(self) -> None:
        super().__init__()
        self._worker_warmed = False

    def warmup_worker(self) -> None:
        """Mark this worker-local simulation instance as warmed."""
        self._worker_warmed = True

    def run(self, context: SimulationContext) -> None:
        """Require warmup before normal mock execution."""
        if not self._worker_warmed:
            raise RuntimeError("worker was not warmed")
        super().run(context)


class FailingWarmupMockSimulation(MockSimulation):
    """Mock backend whose worker warmup always fails."""

    def warmup_worker(self) -> None:
        """Raise a deterministic worker-warmup failure."""
        raise RuntimeError("warmup failed")


class ExtraStatsMockSimulation(MockSimulation):
    """Mock backend that emits one declared extra stat."""

    @property
    def extra_stat_fields(self) -> Mapping[str, u.UnitBase]:
        """Return the declared mock extra stat registry."""
        return {"halo": u.mas}

    def build_extra_stats(self, context: SimulationContext) -> Mapping[str, Any]:
        """Build deterministic per-science extra stats for one simulation."""
        num_sci = int(np.asarray(context.setup.sci_r, dtype=float).reshape(-1).shape[0])
        return {"halo": np.full((num_sci,), float(context.index + 10), dtype=np.float32) * u.mas}


class FailOnceMockSimulation(MockSimulation):
    """Mock backend that fails one selected index once using a marker file."""

    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Load the marker path used to coordinate retry behavior."""
        marker_path = simulation_payload.get("marker_path")
        if not isinstance(marker_path, str) or not marker_path:
            raise ValueError("FailOnceMockSimulation requires a non-empty marker_path.")
        self._marker_path = Path(marker_path)

    def run(self, context: SimulationContext) -> None:
        """Fail index one until the marker file has been created."""
        if int(context.index) == 1 and not self._marker_path.exists():
            self._marker_path.write_text("failed\n", encoding="utf-8")
            raise RuntimeError("intentional one-time failure")
        super().run(context)
