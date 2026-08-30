"""Command-line interface for ao-predict workflows.

This module defines the ``ao-predict simulate ...`` command tree. The
simulation commands bridge YAML/CSV inputs and flags into the code-first
simulation API.
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml
from astropy import units as u

from . import __version__
from .simulation import schema
from .simulation.config import normalize_table_options_config
from .simulation import SimulationState
from .simulation.api import (
    DatasetConfigMismatchError,
    InitDatasetRequest,
    TableOptionsConfig,
    check_dataset,
    init_dataset,
    reset_simulations,
    resume_simulations,
    run_simulations_by_state,
    validate_dataset_matches_request,
)


# YAML/config parsing helpers

def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file as a mapping.

    Args:
        path: YAML file path.

    Returns:
        Parsed top-level mapping. Empty files yield ``{}``.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must contain a top-level mapping/object.")
    return data


def _lowercase_keys_recursive(value: Any) -> Any:
    """Recursively lowercase mapping keys for YAML/CSV user inputs."""
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            norm_key = key.lower() if isinstance(key, str) else key
            out[norm_key] = _lowercase_keys_recursive(item)
        return out
    if isinstance(value, list):
        return [_lowercase_keys_recursive(item) for item in value]
    return value


def _decode_quantity_values(value: Any, *, path: str = "config") -> Any:
    """Decode YAML ``{value, unit}`` mappings into Astropy quantities."""
    if isinstance(value, dict):
        if set(value) == {"value", "unit"}:
            try:
                return u.Quantity(value["value"], unit=u.Unit(value["unit"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} must contain a valid quantity value and unit.") from exc
        return {
            key: _decode_quantity_values(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_quantity_values(item, path=f"{path}[]") for item in value]
    return value


def _parse_table_from_csv(path: str) -> tuple[list[str], list[list[Any]]]:
    """Parse options table input from a CSV file.

    Args:
        path: CSV file path.

    Returns:
        Tuple ``(columns, rows)`` with lowercase column names.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE_PATH} CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = [str(c).lower() for c in (reader.fieldnames or []) if c is not None]
        if not columns:
            raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE_PATH} CSV must include a header row.")
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE_PATH} CSV headers must be unique after lowercasing."
            )
        rows: list[list[Any]] = []
        for rec in reader:
            lower_rec = {str(k).lower(): v for k, v in rec.items() if k is not None}
            rows.append([lower_rec.get(col) for col in columns])
    return columns, rows


def _prepare_options_config(options_cfg: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Normalize raw YAML ``options`` into ``broadcast/columns/rows`` format.

    Args:
        options_cfg: Raw ``options`` mapping from YAML.
        config_dir: Directory containing the YAML file.

    Returns:
        Normalized options config consumed by the API layer.
    """
    table_cfg = options_cfg.get(schema.KEY_CFG_OPTION_TABLE)
    if isinstance(table_cfg, dict):
        columns = table_cfg.get(schema.KEY_CFG_OPTION_COLUMNS)
        if isinstance(columns, list):
            table_cfg[schema.KEY_CFG_OPTION_COLUMNS] = [str(col).lower() if isinstance(col, str) else col for col in columns]

    normalized = normalize_table_options_config(options_cfg)
    table_path = table_cfg.get(schema.KEY_CFG_OPTION_TABLE_PATH) if isinstance(table_cfg, dict) else None
    if table_path is not None:
        if not isinstance(table_path, str):
            raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE_PATH} must be a string path.")
        csv_path = Path(table_path)
        if not csv_path.is_absolute():
            csv_path = config_dir / csv_path
        table_columns, table_rows = _parse_table_from_csv(str(csv_path))
        normalized[schema.KEY_CFG_OPTION_COLUMNS] = table_columns
        normalized[schema.KEY_CFG_OPTION_ROWS] = table_rows
    return normalized


def _load_config(config_yaml: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and normalize ``simulation/setup/options`` from a YAML config file.

    Args:
        config_yaml: YAML configuration path.

    Returns:
        Tuple ``(simulation_cfg, setup_cfg, options_cfg)``.
    """
    def _as_mapping(value: Any, label: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"'{label}' must be a mapping/object.")
        return dict(value)

    config_path = Path(config_yaml)
    config_dir = config_path.parent
    cfg = _decode_quantity_values(_lowercase_keys_recursive(_load_yaml(config_yaml)))
    simulation_cfg = _as_mapping(cfg.get(schema.KEY_SIMULATION_SECTION), schema.KEY_SIMULATION_SECTION)
    simulation_cfg.setdefault(schema.KEY_CFG_SIMULATION_BASE_PATH, str(config_dir))
    setup_cfg = _as_mapping(cfg.get(schema.KEY_SETUP_SECTION), schema.KEY_SETUP_SECTION)
    raw_options_cfg = _as_mapping(cfg.get(schema.KEY_OPTION_SECTION), schema.KEY_OPTION_SECTION)
    options_cfg = _prepare_options_config(raw_options_cfg, config_dir)
    return simulation_cfg, setup_cfg, options_cfg


def _load_init_request(
    config_yaml: str,
    dataset_path: str,
    *,
    overwrite: bool = False,
    save_psfs: bool = False,
) -> InitDatasetRequest:
    """Load a CLI config into an initialization request."""
    simulation_cfg, setup_cfg, options_cfg = _load_config(config_yaml)
    return InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=simulation_cfg,
        setup=setup_cfg,
        options=TableOptionsConfig(
            broadcast=dict(options_cfg.get(schema.KEY_CFG_OPTION_BROADCAST, {})),
            columns=options_cfg.get(schema.KEY_CFG_OPTION_COLUMNS),
            units=dict(options_cfg.get(schema.KEY_CFG_OPTION_UNITS, {})),
            rows=options_cfg.get(schema.KEY_CFG_OPTION_ROWS),
        ),
        overwrite=overwrite,
        save_psfs=save_psfs,
    )


def _parse_index_list(raw_values: str) -> list[int]:
    """Parse comma-separated 1-based simulation numbers into 0-based indexes.

    Args:
        raw_values: Comma-separated simulation numbers.

    Returns:
        Zero-based simulation indexes.
    """
    parts = [p.strip() for p in raw_values.split(",")]
    if any(p == "" for p in parts):
        raise ValueError("selection list must be a comma-separated list of integers (e.g. 1,4,7).")
    out: list[int] = []
    for raw in parts:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid index '{raw}' in selection list.") from exc
        if value <= 0:
            raise ValueError("--sims values must be >= 1.")
        out.append(value - 1)
    return out


def _resolve_selected_indexes(args: argparse.Namespace) -> list[int] | None:
    """Resolve optional ``--sims`` selection into API indexes.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Zero-based selected indexes or ``None``.
    """
    if getattr(args, "sims", None) is not None:
        return _parse_index_list(args.sims)
    return None


# Command handlers

def _handle_simulate_init(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate init`` command."""
    if args.dataset is not None:
        dataset_path = str(args.dataset)
    else:
        cfg_path = Path(args.config_yaml)
        dataset_path = str(cfg_path.parent / "sims" / f"{cfg_path.stem}.h5")
    num_sims = init_dataset(
        _load_init_request(
            args.config_yaml,
            dataset_path,
            overwrite=bool(args.overwrite),
            save_psfs=bool(args.save_psfs),
        )
    )

    print(f"Initialized dataset: {dataset_path}")
    print(f"Simulations: {num_sims}")
    return 0


def _handle_simulate_run(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate run`` command."""
    indexes = _resolve_selected_indexes(args)
    summary = run_simulations_by_state(
        args.dataset,
        state=SimulationState.PENDING,
        verbose=bool(args.verbose),
        indexes=indexes,
        num_workers=int(args.threads),
        chunk_multiple=int(args.chunks),
    )

    print(
        f"Run summary: attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    return 0


def _handle_simulate_retry(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate retry`` command."""
    indexes = _resolve_selected_indexes(args)
    summary = run_simulations_by_state(
        args.dataset,
        state=SimulationState.FAILED,
        verbose=bool(args.verbose),
        indexes=indexes,
        num_workers=int(args.threads),
        chunk_multiple=int(args.chunks),
    )

    print(
        f"Retry summary: attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    return 0


def _handle_simulate_resume(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate resume`` command."""
    expected_request = None
    if args.config is not None:
        expected_request = _load_init_request(args.config, args.dataset)

    summary = resume_simulations(
        args.dataset,
        expected_request=expected_request,
        verbose=bool(args.verbose),
        num_workers=int(args.threads),
        chunk_multiple=int(args.chunks),
    )

    print(
        f"Resume summary: attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    return 0


def _handle_simulate_check(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate check`` command."""
    status = check_dataset(args.dataset)
    config_mismatch: DatasetConfigMismatchError | None = None
    config_error: ValueError | None = None
    if args.config is not None:
        try:
            validate_dataset_matches_request(
                args.dataset,
                _load_init_request(args.config, args.dataset),
            )
        except DatasetConfigMismatchError as exc:
            config_mismatch = exc
        except ValueError as exc:
            config_error = exc

    if not status.ok or config_mismatch is not None or config_error is not None:
        print(f"Dataset check FAILED: {args.dataset}")
        for issue in status.issues:
            print(f"- {issue}")
        if config_mismatch is not None:
            for mismatch in config_mismatch.mismatches:
                print(f"- Config mismatch: {mismatch}")
        if config_error is not None:
            print(f"- Config validation failed: {config_error}")
        return 1

    print(f"Dataset check PASSED: {args.dataset}")
    print(f"All simulations completed successfully (N={status.num_sims}).")
    return 0


def _handle_simulate_reset(args: argparse.Namespace) -> int:
    """Handle ``ao-predict simulate reset`` command."""
    indexes = _resolve_selected_indexes(args)
    changed = reset_simulations(args.dataset, indexes=indexes)
    print(f"Reset summary: changed={changed}")
    return 0


# Parser construction

def _build_parser() -> argparse.ArgumentParser:
    """Build root CLI parser."""
    parser = argparse.ArgumentParser(
        prog="ao-predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="mode", metavar="mode", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="simulation mode commands")
    simulate_subparsers = simulate_parser.add_subparsers(dest="mode_command", metavar="command", required=True)

    simulate_command_help: dict[str, str] = {}

    simulate_command_help["init"] = "initialize simulation dataset"
    simulate_init_parser = simulate_subparsers.add_parser("init", help=simulate_command_help["init"])
    simulate_init_parser.add_argument("config_yaml", help="single simulation config YAML path")
    simulate_init_parser.add_argument("--dataset", help="dataset HDF5 path (default: inferred from config_yaml path)")
    simulate_init_parser.add_argument("--overwrite", action="store_true", help="overwrite dataset if it exists")
    simulate_init_parser.add_argument("--save-psfs", action="store_true", help="persist PSF cubes in /psfs/data")

    simulate_command_help["run"] = "run all pending simulations"
    simulate_run_parser = simulate_subparsers.add_parser("run", help=simulate_command_help["run"])
    simulate_run_parser.add_argument("dataset", help="dataset HDF5 path")
    simulate_run_parser.add_argument("--sims", help="optional comma-separated simulation numbers to run (1-based)")
    simulate_run_parser.add_argument("--threads", type=int, default=1, help="number of parallel joblib workers")
    simulate_run_parser.add_argument("--chunks", type=int, default=10, help="simulations assigned to each worker chunk")
    simulate_run_parser.add_argument("--verbose", action="store_true", help="print failure messages for failed simulations")

    simulate_command_help["retry"] = "retry all failed simulations"
    simulate_retry_parser = simulate_subparsers.add_parser("retry", help=simulate_command_help["retry"])
    simulate_retry_parser.add_argument("dataset", help="dataset HDF5 path")
    simulate_retry_parser.add_argument("--sims", help="optional comma-separated simulation numbers to retry (1-based)")
    simulate_retry_parser.add_argument("--threads", type=int, default=1, help="number of parallel joblib workers")
    simulate_retry_parser.add_argument("--chunks", type=int, default=10, help="simulations assigned to each worker chunk")
    simulate_retry_parser.add_argument("--verbose", action="store_true", help="print failure messages for failed simulations")

    simulate_command_help["resume"] = "run pending simulations and preexisting failures"
    simulate_resume_parser = simulate_subparsers.add_parser("resume", help=simulate_command_help["resume"])
    simulate_resume_parser.add_argument("dataset", help="dataset HDF5 path")
    simulate_resume_parser.add_argument("--config", help="optional initialization config YAML expected to match the dataset")
    simulate_resume_parser.add_argument("--threads", type=int, default=1, help="number of parallel joblib workers")
    simulate_resume_parser.add_argument("--chunks", type=int, default=10, help="simulations assigned to each worker chunk")
    simulate_resume_parser.add_argument("--verbose", action="store_true", help="print failure messages for failed simulations")

    simulate_command_help["reset"] = "reset all simulations to pending state"
    simulate_reset_parser = simulate_subparsers.add_parser("reset", help=simulate_command_help["reset"])
    simulate_reset_parser.add_argument("dataset", help="dataset HDF5 path")
    simulate_reset_parser.add_argument("--sims", help="optional comma-separated simulation numbers (1-based) to reset (default: all simulations)")

    simulate_command_help["check"] = "validate dataset schema and completion status"
    simulate_check_parser = simulate_subparsers.add_parser("check", help=simulate_command_help["check"])
    simulate_check_parser.add_argument("dataset", help="dataset HDF5 path")
    simulate_check_parser.add_argument("--config", help="optional initialization config YAML expected to match the dataset")

    parser.epilog = "simulate commands:\n" + "\n".join(
        f"  {name:<6} {help_text}" for name, help_text in simulate_command_help.items()
    )

    return parser


# CLI entry point

def main() -> int:
    """CLI entry point.

    Returns:
        Process-style exit code. On argument parsing errors argparse raises
        ``SystemExit`` with code 2.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "simulate":
        if args.mode_command == "init":
            return _handle_simulate_init(args)
        if args.mode_command == "run":
            return _handle_simulate_run(args)
        if args.mode_command == "retry":
            return _handle_simulate_retry(args)
        if args.mode_command == "resume":
            return _handle_simulate_resume(args)
        if args.mode_command == "reset":
            return _handle_simulate_reset(args)
        if args.mode_command == "check":
            return _handle_simulate_check(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
