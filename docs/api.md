# Python API Documentation

This document describes the primary code-first simulation API exposed at
`ao_predict` and implemented in `ao_predict.simulation.api`.

For analysis reads from an existing dataset, use
`ao_predict.load_analysis_dataset(...)` or
`ao_predict.analysis.load_analysis_dataset(...)`. That is the supported
upstream analysis read path; callers should not need to construct
`SimulationStore` directly just to load analysis views.

For Matplotlib PSF, PSF-core, and metric-field plots from loaded analysis
simulations, use `ao_predict.plotting`. Plotting helpers live in that submodule
rather than the package root.

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
- Successful runs may persist `fwhm_mas = NaN` when contour-based FWHM cannot be
  recovered; `sr` and `ee` remain finite for successful results.
- Dataset-level stats selectors live under `/setup` as `sr_method`, `fwhm_summary`, and `ee_geometry`.
- The implemented core stats family is:
  - Strehl: image-domain `pixel_fit` (default) or `pixel_max`
  - EE: fixed peak-centered image-domain aperture accumulation selected by `/setup/ee_geometry`
  - FWHM: fixed native contour measurement summarized by `/setup/fwhm_summary`
- Core metadata under `/meta` mixes one per-simulation field and invariant telescope fields:
  - `/meta/pixel_scale_mas` is `[N]`
  - `/meta/tel_diameter_m` is a scalar
  - `/meta/tel_pupil` is `[Ny, Nx]`
- Simulations may also declare extra 2D stats with shape `[N, M]`.
- The declared extra stat registry is persisted in `/simulation/extra_stat_names`.
- During execution, successful simulations expose PSFs and metadata in `finalize(...)` and leave `result.stats` empty.
- ao-predict computes the core stats from PSFs and assembles the final `result.stats`.
- Simulations contribute only declared extra stats through the `Simulation.build_extra_stats(...)` hook.

### `run_simulations_by_state(dataset_path: str | Path, *, state: SimulationState | int = SimulationState.PENDING, verbose: bool = False, indexes: list[int] | None = None, num_workers: int = 1, chunk_multiple: int = 10) -> RunSummary`
Run simulations for a selected source state.

Supported `state` values:
- `SimulationState.PENDING`: run pending simulations
- `SimulationState.FAILED`: retry failed simulations

Execution controls:
- `num_workers=1` runs serially and preserves the original execution behavior.
- `num_workers > 1` uses joblib/loky worker processes.
- `chunk_multiple` controls how many simulations are assigned to each worker chunk.
- Worker processes return completed in-memory results; HDF5 writes remain owned by the parent process.
- Simulation implementations may override `Simulation.warmup_worker()` to prepare process-local worker state before a chunk runs.

### `resume_simulations(dataset_path: str | Path, *, expected_request: InitDatasetRequest | None = None, verbose: bool = False, num_workers: int = 1, chunk_multiple: int = 10) -> RunSummary`
Resume a dataset by running pending rows and retrying only preexisting failures.

Behavior:
- validates schema
- validates the dataset against `expected_request` when supplied
- records rows that are failed before the call begins
- runs pending rows
- retries only rows that were already failed before the call began
- does not retry newly failed rows in the same invocation

`save_psfs` is not a resume option. Resumed rows use the storage layout chosen when the dataset was initialized.

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

### `validate_dataset_matches_request(dataset_path: str | Path, request: InitDatasetRequest) -> None`
Validate that an existing dataset matches an initialization request.

The request is prepared through the same lifecycle as `init_dataset(...)`.
The prepared `/simulation`, `/setup`, and `/options` payloads are compared to
the persisted dataset payloads. The request's `dataset_path`, `overwrite`, and
`save_psfs` fields do not participate in matching.

Raises:
- `DatasetConfigMismatchError` when prepared payload values differ from the existing dataset.

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
- `ee_geometry: str | None = None`
- `specific_fields: dict[str, object] = {}`

Core typed setup fields are `ee_apertures_mas`, `sr_method`, `fwhm_summary`, and `ee_geometry`. All other setup fields can be passed in `specific_fields`.
For `TiptopSimulation`, include `specific_fields["ngs_mag_zeropoint"]`.
These setup fields control how persisted `/stats/sr`, `/stats/ee`, and
`/stats/fwhm_mas` are computed and interpreted across the whole dataset.

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
- `wavelength_um` is required at execution time because the core Strehl
  calculation builds a diffraction-limited reference PSF for each simulation.
- The persisted `/options` payload always contains the NGS triplet (`ngs_r_arcsec`, `ngs_theta_deg`, `ngs_mag`).
- If NGS input is provided explicitly, provide the full triplet. Unused star slots may be represented with `NaN`, but each slot must be either all finite or all `NaN` across the triplet.
- If explicit NGS input is omitted, the selected simulation must supply the persisted NGS triplet during options preparation.
- During execution, ao-predict derives a runtime-only `ngs_used` boolean vector from the persisted NGS triplet. This field is not persisted in `/options`.
- If omitted, setup defaults `sr_method` to `pixel_fit`, `fwhm_summary` to `geom`, and `ee_geometry` to `ensquared`.

Atmospheric input note:
- `r0_m` is the canonical persisted per-sim atmospheric option.
- `seeing_arcsec` is accepted as an input alias and converted to `r0_m` using `setup.atm_wavelength_um` before persistence.
- `seeing_arcsec` is never persisted in `/options`.
- In `setup.atm_profiles`, `seeing_arcsec` is accepted per profile and normalized to `r0_m` before persistence.
- Bound `SimulationSetup` instances contain normalized concrete arrays for `lgs_*` and `sci_*` fields; absent LGS inputs are represented as empty arrays, not `None`.

Generated table helper:
- Use `ao_predict.simulation.options.options_from_rows(...)` when code
  generates one mapping per simulation and needs a stable
  `TableOptionsConfig`.
- The helper returns `GeneratedOptions`, which preserves table `columns`,
  `rows`, and `broadcast` defaults and exposes
  `to_table_options_config()` for `InitDatasetRequest.options`.

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
    resume_simulations,
    run_simulations_by_state,
    validate_dataset_matches_request,
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
        ee_geometry="ensquared",
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
resume_summary = resume_simulations(dataset_path, expected_request=request)
validate_dataset_matches_request(dataset_path, request)
status = check_dataset(dataset_path)

print(summary)
print(resume_summary)
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

## Interpolation API

`ao_predict.interpolation` owns the generic interpolation artifact contracts.
Project code owns conversion from native simulation products into
`NgsHoPsfSamples`, `NgsHoMetricSamples`, or `ScienceHoPsfSamples`.

The example below builds the two interpolation artifacts consumed by
`HybridSimulation` and passes their paths into `SimulationConfig`. Replace the
placeholder arrays with physically valid processed HO simulation products before
building production artifacts.

Physical axes are optional when fixed. AO Predict stores fixed physical values
as metadata and makes only multi-value axes active interpolation coordinates.
Wavelength is PSF-stat metadata for NGS-HO metrics; it is not an NGS
interpolation coordinate.

### HybridSimulation Interpolator Artifacts

```python
from pathlib import Path
import numpy as np
import ao_predict as aop
import ao_predict.interpolation as interp

x_arcsec = np.asarray([-30.0, 0.0, 30.0, -30.0, 0.0, 30.0, -30.0, 0.0, 30.0], dtype=float)
y_arcsec = np.asarray([-30.0, -30.0, -30.0, 0.0, 0.0, 0.0, 30.0, 30.0, 30.0], dtype=float)
ngs_psfs = np.ones((x_arcsec.size, 100, 100), dtype=np.float32) / 100**2
science_psfs = np.ones((x_arcsec.size, 100, 100), dtype=np.float32) / 100**2
tel_diameter_m = 8.0
tel_pupil = np.ones((64, 64), dtype=np.float32)

ngs_samples = interp.NgsHoPsfSamples(
    x_arcsec=x_arcsec,
    y_arcsec=y_arcsec,
    psfs=ngs_psfs,
    wavelength_um=0.710,
    pixel_scale_mas=5.0,
    tel_diameter_m=tel_diameter_m,
    tel_pupil=tel_pupil,
)

ngs_interpolator_path = Path("ngs_ho_metric_interpolator.pkl")
interp.save_ngs_ho_metric_interpolator(
    interp.build_ngs_ho_metric_interpolator_from_psfs(ngs_samples),
    ngs_interpolator_path,
)

science_samples = interp.ScienceHoPsfSamples(
    x_arcsec=x_arcsec,
    y_arcsec=y_arcsec,
    psfs=science_psfs,
    wavelength_um=2.179,
    pixel_scale_mas=5.0,
    tel_diameter_m=tel_diameter_m,
    tel_pupil=tel_pupil,
)

science_interpolator_path = Path("science_ho_psf_interpolator.pkl")
interp.save_science_ho_psf_interpolator(
    interp.build_science_ho_psf_interpolator(science_samples),
    science_interpolator_path,
)

hybrid_simulation = aop.SimulationConfig(
    name="Hybrid",
    base_path=".",
    specific_fields={
        "config_path": "mastsel.ini",
        "science_ho_psf_interpolator_path": str(science_interpolator_path),
        "ngs_ho_metric_interpolator_path": str(ngs_interpolator_path),
    },
)
```

To make zenith/airmass active, provide `zenith_angle_deg` with one value per
source plane. For Science-HO wavelength or zenith/airmass interpolation, supply
per-plane `wavelength_um`, `pixel_scale_mas`, and optionally `zenith_angle_deg`,
and shape `psfs` as `(planes, field points, psf_y, psf_x)`. For smoothed NGS
metric fields, pass `RbfInterpolationConfig(...)` to
`build_ngs_ho_metric_interpolator_from_psfs(...)`.

## Error Behavior
- Invalid payload structure and schema mismatches raise `ValueError`/`TypeError`.
- Existing dataset without `overwrite=True` raises `FileExistsError`.
- Dataset/config mismatches raise `DatasetConfigMismatchError`.


## PSF Stats API

`compute_psf_stats(...)` computes AO Predict core PSF statistics from one
PSF image or a PSF cube. Import it from the package root together with
`PsfMetadata`, focused metric helpers, and the named preprocessing helper when
needed:

```python
from ao_predict import (
    PsfMetadata,
    clip_and_sum_normalize_psfs,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_stats,
)

metadata = PsfMetadata(
    wavelength_um=1.65,
    pixel_scale_mas=4.0,
    tel_diameter_m=8.0,
    tel_pupil=tel_pupil,
)

sr, ee, fwhm_mas = compute_psf_stats(
    psfs,
    metadata,
    ee_apertures_mas=[50.0, 100.0],
    sr_method="pixel_fit",
    fwhm_summary="geom",
    ee_geometry="ensquared",
    preprocess=clip_and_sum_normalize_psfs,
)

fwhm_only = compute_psf_fwhm(
    psfs,
    metadata,
    fwhm_summary="geom",
    preprocess="default",
)

ee_only = compute_psf_ee(
    psfs,
    metadata,
    ee_apertures_mas=[50.0, 100.0],
    sr_method="pixel_fit",
    ee_geometry="ensquared",
    preprocess="default",
)

selected = compute_psf_stats(
    psfs,
    metadata,
    metrics=("fwhm_mas", "ee"),
    ee_apertures_mas=[50.0, 100.0],
    preprocess="default",
)
```

`wavelength_um` and `pixel_scale_mas` may be scalar values shared by all PSFs,
or one-dimensional per-PSF arrays matching the PSF cube length. `ee_apertures_mas`
may be a shared aperture vector or a per-PSF aperture array with shape `[M, A]`.
It is required only when enclosed energy is computed.

When `metrics` is omitted, `compute_psf_stats(...)` returns the legacy tuple
`(sr, ee, fwhm_mas)`. When `metrics` is supplied, it must be a non-empty
sequence drawn from `"sr"`, `"ee"`, and `"fwhm_mas"`; the return value is a
tuple in the requested order. The focused helpers
`compute_psf_sr(...)`, `compute_psf_ee(...)`, and `compute_psf_fwhm(...)` return
only their named metric and use the same metadata, selector, and preprocessing
contracts.

Metric selector options are:

- `sr_method`: `"pixel_fit"` by default; also supports `"pixel_max"`.
- `fwhm_summary`: `"geom"` by default; also supports `"mean"`, `"max"`, and `"min"`.
- `ee_geometry`: `"ensquared"` by default; also supports `"encircled"`.

The public stats function does no clipping, centering, or normalization unless
`preprocess` is supplied. External callers can omit `preprocess` when PSFs are
already metric-ready, pass `clip_and_sum_normalize_psfs` or `preprocess="default"`
to use AO Predict's shared non-negative clipping and pixel-sum normalization
path, or pass another callable with signature `preprocess(psfs)`.

## Plotting Helpers

`ao_predict.plotting` provides Matplotlib helpers for loaded analysis
simulations:

- `plot_psf(sim, psf_index=0)`
- `plot_psf_core(sim, psf_index=0)`
- `plot_metric_field(sim, metric_name="sr")`
- `plot_metric_field_panel([sim1, sim2, ...], metric_name="sr")`
- `plot_metric_field_comparison(left, right, metric_name="sr")`
- `prepare_metric_field_grid(...)`
- `prepare_metric_field_comparison_grid(...)`

They return unshown Matplotlib figures. PSF helpers use the persisted pixel
scale for milliarcsecond axes. Metric-field plots use persisted science
coordinates and generic SciPy interpolation to render a regular field image.
Metric-field plots can also draw generic NGS/LGS markers from persisted
coordinates and optional contour overlays when requested.

Metric-field panel and comparison helpers own generic subplot layout, shared
colorbar placement, metric-name placement, relative-percent comparison fields,
and prepared grid rows for dense caller-composed figures. They accept
`field_plotter` and `field_plotter_kwargs` so downstream packages can reuse AO
Predict composition while routing each field through a domain-specific
single-field wrapper.
- `check_dataset` returns issues in `DatasetStatus` for schema/state problems instead of raising, unless the file cannot be opened/read at all.

## Analysis Read Path

`load_analysis_dataset(path, *, dataset_cls=AnalysisDataset, simulation_cls=AnalysisSimulation, extra_field_extractors=None)`
returns an immutable `AnalysisDataset` built from store-backed reads after
schema validation succeeds.

Use:
- `len(dataset)` for the dataset size
- `dataset.setup` for dataset-level setup values
- `dataset.options` for eager per-simulation option columns
- `dataset.meta` for eager loaded-analysis metadata values
- `dataset.stats` for eager per-simulation stats columns
- `dataset.sim(i)` to get one `AnalysisSimulation`
- `sim.config` for the persisted simulation view with exactly `setup` and `options`
- `sim.meta` for persisted scientific metadata, including per-simulation
  `pixel_scale_mas` plus dataset-level telescope metadata such as
  `tel_diameter_m` and `tel_pupil`
- `sim.stats` for the persisted scientific stats surface, including core
  `sr`, `ee`, and `fwhm_mas` plus any declared extra stats
- `sim.psfs` for lazy PSF access when PSFs were saved

If the dataset was initialized without persisted PSFs, `sim.psfs` raises a
clear error instead of falling back silently.

The loader also exposes a generic downstream-agnostic extension seam.
Downstream packages can override `from_load_payload(...)` on dataset/simulation
subclasses and register `extra_field_extractors` that read through
`AnalysisLoadContext` and return `AnalysisLoadContribution` objects. Those
contributions can add eager or lazy dataset/simulation fields without
re-loading the dataset or exposing raw HDF5 handles, and downstream simulation
properties can access them through `_require_extra_field(...)`.

`AnalysisDataset` is generic over the simulation view type, so downstream
dataset subclasses can declare their simulation subtype directly and usually
inherit `sim()` unchanged:

```python
from ao_predict.analysis import AnalysisDataset, AnalysisSimulation


class CustomAnalysisSimulation(AnalysisSimulation):
    ...


class CustomAnalysisDataset(AnalysisDataset[CustomAnalysisSimulation]):
    pass
```

Compatibility adapters and legacy shaping are handled by `girmos-aosims`.

## Current Limits
- Execution mode is serial.
- Dataset path is required.
- Parallel workers, automatic option generation, and high-level data-loading utilities are not yet implemented.
