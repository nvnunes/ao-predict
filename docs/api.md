# Python API Documentation

This document describes the primary code-first simulation API exposed at
`ao_predict` and implemented in `ao_predict.simulation.api`.

## Lifecycle Functions

### `init_dataset(request: InitDatasetRequest) -> int`
Initialize an HDF5 simulation dataset from code-provided config.

Responsibilities:
- create and validate simulation payload (`/simulation`)
- create and validate setup payload (`/setup`)
- complete and validate options payload (`/options`)
- allocate status/meta/stats (and optional psfs) datasets

Simulation payload note:
- ao-predict assembles the core `/simulation` fields: `name`, `version`, and `extra_stat_names`.
- Simulations expose the extra-stat registry through the `Simulation.extra_stat_names` property.
- The simulation implementation completes that base payload with simulation-specific persisted fields.
- This mirrors the existing core-plus-completion pattern used for `/setup` and `/options`.

Stats note:
- Core stats under `/stats` are `sr`, `ee`, and `fwhm_mas`.
- Dataset-level stats selectors live under `/setup` as `sr_method` and `fwhm_summary`.
- Simulations may also declare extra 2D stats with shape `[N, M]`.
- The declared extra stat registry is persisted in `/simulation/extra_stat_names`.
- During execution, successful simulations expose PSFs and metadata in `finalize(...)` and leave `result.stats` empty.
- ao-predict computes the core stats from PSFs and assembles the final `result.stats`.
- Simulations contribute only declared extra stats through the `Simulation.build_extra_stats(...)` hook.

### `run_simulations_by_state(dataset_path: str | Path, *, state: SimulationState | int = SimulationState.PENDING, verbose: bool = False, indexes: list[int] | None = None) -> RunSummary`
Run simulations for a selected source state.

Supported `state` values:
- `SimulationState.PENDING`: run pending simulations
- `SimulationState.FAILED`: retry failed simulations

### `reset_simulations(dataset_path: str | Path, indexes: list[int] | None = None) -> int`
Reset all simulations to pending state (`SimulationState.PENDING`).

Returns:
- number of simulations whose state changed

Notes:
- If `indexes` is provided, only those simulation indexes are reset.
- Existing `/stats`, `/meta`, and `/psfs` values are retained and overwritten as simulations are rerun.

### `check_dataset(dataset_path: str | Path) -> DatasetStatus`
Validate schema and completion status.

`ok=True` only when:
- schema validation passes
- `num_pending == 0`
- `num_failed == 0`

### `validate_dataset(dataset_path: str | Path) -> None`
Strict dataset validation that raises when issues are present.

Raises:
- `DatasetValidationError` when schema/state checks fail.

## Dataclasses

### `SimulationConfig`
- `name: str`
- `base_path: str | None = None`
- `specific_fields: dict[str, object] = {}`

Use `specific_fields` for simulation-specific passthrough keys.
For `TiptopSimulation`, provide `specific_fields["config_path"]` and optionally
`base_path` to resolve relative `config_path` values.

### `SetupConfig`
- `ee_apertures_mas: list[float]`
- `sr_method: str | None = None`
- `fwhm_summary: str | None = None`
- `specific_fields: dict[str, object] = {}`

Core typed setup fields are `ee_apertures_mas`, `sr_method`, and `fwhm_summary`. All other setup fields can be passed in `specific_fields`.
For `TiptopSimulation`, include `specific_fields["ngs_mag_zeropoint"]`.

### `OptionsConfig`
- `option_arrays: dict[str, np.ndarray | list[object] | tuple[object, ...]]`

Columnar per-option arrays keyed by option names.

### `TableOptionsConfig`
- `broadcast: dict[str, object] = {}`
- `columns: list[str] | None = None`
- `rows: list[list[object]] | None = None`

Config-style options input for table/broadcast workflows.

### `InitDatasetRequest`
- `dataset_path: str | Path`
- `simulation: SimulationConfig | Mapping[str, object]`
- `setup: SetupConfig | Mapping[str, object]`
- `options: OptionsConfig | TableOptionsConfig | Mapping[str, np.ndarray | list[object] | tuple[object, ...]]`
- `overwrite: bool = False`
- `save_psfs: bool = False`

### `DatasetStatus`
- `dataset_path: Path`
- `num_sims: int`
- `num_pending: int`
- `num_failed: int`
- `num_succeeded: int`
- `ok: bool`
- `issues: list[str]`

### `DatasetValidationError`
- subclass of `ValueError`
- `issues: list[str]` with collected validation messages

## Options Input Modes

`init_dataset` supports three options payload styles:

1. `OptionsConfig(option_arrays=...)` typed columnar input.
2. `TableOptionsConfig(...)` typed table/broadcast input.
3. Raw direct columnar mapping (`{key: ndarray}`).

Notes:
- Inputs must be columnar per-option arrays with first dimension `N` (one entry per simulation).
- Use columnar arrays when calling `init_dataset` from Python code.
- API mapping keys are case-sensitive and must be lowercase (`simulation`, `setup`, and options keys).
- `TableOptionsConfig.columns` and `TableOptionsConfig.broadcast` keys must be lowercase.
- The persisted `/options` payload always contains the NGS triplet (`ngs_r_arcsec`, `ngs_theta_deg`, `ngs_mag`).
- If NGS input is provided explicitly, provide the full triplet. Unused star slots may be represented with `NaN`, but each slot must be either all finite or all `NaN` across the triplet.
- If explicit NGS input is omitted, the selected simulation must supply the persisted NGS triplet during options preparation.
- During execution, ao-predict derives a runtime-only `ngs_used` boolean vector from the persisted NGS triplet. This field is not persisted in `/options`.
- If omitted, setup defaults `sr_method` to `pixel_fit` and `fwhm_summary` to `geom`.

Atmospheric input note:
- `r0_m` is the canonical persisted per-sim atmospheric option.
- `seeing_arcsec` is accepted as an input alias and converted to `r0_m` using `setup.atm_wavelength_um` before persistence.
- `seeing_arcsec` is never persisted in `/options`.
- In `setup.atm_profiles`, `seeing_arcsec` is accepted per profile and normalized to `r0_m` before persistence.
- Bound `SimulationSetup` instances contain normalized concrete arrays for `lgs_*` and `sci_*` fields; absent LGS inputs are represented as empty arrays, not `None`.

## Working Example

```python
from pathlib import Path

from ao_predict import (
    InitDatasetRequest,
    SetupConfig,
    SimulationConfig,
    SimulationState,
    TableOptionsConfig,
    check_dataset,
    init_dataset,
    run_simulations_by_state,
)

request = InitDatasetRequest(
    dataset_path="examples/sims/demo.h5",
    simulation=SimulationConfig(
        name="Tiptop",
        base_path="examples",
        specific_fields={"config_path": "sample_tiptop.ini"},
    ),
    setup=SetupConfig(
        ee_apertures_mas=[50.0, 100.0],
        sr_method="pixel_fit",
        fwhm_summary="geom",
        specific_fields={"ngs_mag_zeropoint": 3.0e10},
    ),
    options=TableOptionsConfig(
        broadcast={"zenith_angle_deg": 20.0},
        columns=["wavelength_um"],
        rows=[[1.654], [2.179]],
    ),
    overwrite=True,
    save_psfs=False,
)

num_sims = init_dataset(request)
dataset_path = Path(request.dataset_path)
summary = run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
status = check_dataset(dataset_path)

print(summary)
print(num_sims)
print(status.ok, status.issues)
```

See also:
- `examples/simulate_tiptop_api.py`
- `examples/simulate_tiptop_cli_example1.yaml`
- `examples/simulate_tiptop_cli_example2.yaml`
- `examples/simulate_tiptop_cli_example2.csv`
- `examples/simulate_tiptop_cli.sh`
- `examples/sample_tiptop.ini`

## Error Behavior
- Invalid payload structure and schema mismatches raise `ValueError`/`TypeError`.
- Existing dataset without `overwrite=True` raises `FileExistsError`.
- `check_dataset` returns issues in `DatasetStatus` for schema/state problems instead of raising, unless the file cannot be opened/read at all.

## Current Limits
- Execution mode is serial.
- Dataset path is required.
- Parallel workers, automatic option generation, and high-level data-loading utilities are not yet implemented.
