"""Configuration-backed simulation lifecycle base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import schema
from .base import BaseSimulation


class ConfigBackedSimulation(BaseSimulation):
    """Base class for simulations that persist a source configuration file.

    ``ConfigBackedSimulation`` owns the generic lifecycle for reading an input
    ``config_path`` and persisting its text under ``/simulation/base_config``.
    It intentionally does not interpret the configuration format; subclasses
    provide validation and binding hooks for concrete formats such as INI.

    Attributes:
        base_config_text: Persisted source configuration text loaded from a
            simulation payload.
    """

    KEY_SETUP_CONFIG_PATH = "config_path"
    KEY_SETUP_BASE_CONFIG = "base_config"

    def __init__(self) -> None:
        """Initialize unbound configuration-backed simulation state."""
        super().__init__()
        self._base_config_text: str | None = None

    @property
    def base_config_text(self) -> str:
        """Return loaded base configuration text.

        Returns:
            The exact serialized source configuration text from the bound
            ``/simulation/base_config`` payload field.

        Raises:
            TypeError: If ``load_simulation_payload()`` has not been called
                successfully.
        """
        if self._base_config_text is None:
            raise TypeError(f"{type(self).__name__} base config is not configured. Call load_simulation_payload(...) first.")
        return self._base_config_text

    def prepare_simulation_payload(
        self,
        base_simulation_payload: Mapping[str, Any],
        simulation_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build a persisted ``/simulation`` payload from a config file path.

        Args:
            base_simulation_payload: Payload fields prepared by
                ``BaseSimulation`` and owned by the generic simulation
                lifecycle.
            simulation_cfg: User configuration mapping containing
                ``config_path`` and, optionally, ``base_path`` for relative
                path resolution.

        Returns:
            A new simulation payload with ``config_path`` removed and the
            source file text stored under ``base_config``.

        Raises:
            TypeError: If ``config_path`` or ``base_path`` have invalid types.
            ValueError: If ``config_path`` is missing.
            FileNotFoundError: If the resolved configuration file does not
                exist.
        """
        config_path = self._resolve_config_path(simulation_cfg)
        config_text = config_path.read_text(encoding="utf-8")

        simulation_payload = self._build_simulation_payload(
            base_simulation_payload,
            simulation_cfg,
            exclude_keys={self.KEY_SETUP_CONFIG_PATH},
        )
        simulation_payload[self.KEY_SETUP_BASE_CONFIG] = config_text
        return simulation_payload

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate a persisted simulation payload without binding it.

        Args:
            simulation_payload: Candidate persisted ``/simulation`` payload.

        Raises:
            TypeError: If persisted fields have invalid types.
            ValueError: If required persisted fields are missing or invalid.
            Exception: Any format-specific validation error raised by
                ``_prepare_base_config_binding()``.
        """
        super().validate_simulation_payload(simulation_payload)
        _ = self._prepare_base_config_binding(self._get_required_base_config_text(simulation_payload))

    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Bind persisted base configuration text for later lifecycle stages.

        Args:
            simulation_payload: Persisted ``/simulation`` payload containing
                the serialized ``base_config`` text.

        Raises:
            TypeError: If persisted fields have invalid types.
            ValueError: If required persisted fields are missing or invalid.
            Exception: Any format-specific validation or binding error raised
                by subclass hooks.
        """
        base_config_text = self._get_required_base_config_text(simulation_payload)
        base_config = self._prepare_base_config_binding(base_config_text)
        self._base_config_text = base_config_text
        self._bind_base_config(base_config)

    def _resolve_config_path(self, simulation_cfg: Mapping[str, Any]) -> Path:
        """Resolve ``simulation.config_path`` against optional ``simulation.base_path``."""
        source_path = simulation_cfg.get(self.KEY_SETUP_CONFIG_PATH)
        if source_path is None:
            raise ValueError(
                f"{type(self).__name__}.prepare_simulation_payload requires "
                f"simulation['{self.KEY_SETUP_CONFIG_PATH}'] in YAML input."
            )
        if not isinstance(source_path, str):
            raise TypeError(f"simulation['{self.KEY_SETUP_CONFIG_PATH}'] must be a string.")

        config_path = Path(source_path)
        if not config_path.is_absolute():
            base_path = simulation_cfg.get(schema.KEY_CFG_SIMULATION_BASE_PATH)
            if base_path is not None:
                if not isinstance(base_path, str):
                    raise TypeError(f"simulation['{schema.KEY_CFG_SIMULATION_BASE_PATH}'] must be a string when provided.")
                config_path = Path(base_path) / config_path

        if not config_path.is_file():
            raise FileNotFoundError(f"{self._config_file_description()} not found: {config_path}")
        return config_path

    def _get_required_base_config_text(self, simulation_payload: Mapping[str, Any]) -> str:
        """Read required serialized base config text from ``/simulation``."""
        if self.KEY_SETUP_BASE_CONFIG not in simulation_payload:
            raise ValueError(f"{type(self).__name__} requires simulation['{self.KEY_SETUP_BASE_CONFIG}'].")
        base_config_text = simulation_payload[self.KEY_SETUP_BASE_CONFIG]
        if not isinstance(base_config_text, str):
            raise TypeError(f"simulation['{self.KEY_SETUP_BASE_CONFIG}'] must be a string for {type(self).__name__}.")
        return base_config_text

    def _validate_base_config_text(self, base_config_text: str) -> None:
        """Validate loaded base configuration text without mutating state."""
        del base_config_text

    def _prepare_base_config_binding(self, base_config_text: str) -> Any:
        """Validate and prepare subclass-specific base config before binding.

        This hook must not mutate instance state. ``load_simulation_payload()``
        commits ``base_config_text`` and the prepared object only after this
        hook succeeds.
        """
        self._validate_base_config_text(base_config_text)
        return base_config_text

    def _bind_base_config(self, base_config: Any) -> None:
        """Bind prepared base configuration state after validation succeeds."""
        del base_config

    def _config_file_description(self) -> str:
        """Return the user-facing description for missing config-file errors."""
        return f"{type(self).__name__} config file"
