# Python API Documentation

This document describes the primary code-first simulation, model-training,
prediction, and evaluation APIs exposed at `ao_predict`.

For analysis reads from an existing dataset, use
`ao_predict.load_analysis_dataset(...)` or
`ao_predict.analysis.load_analysis_dataset(...)`. That is the supported
upstream analysis read path; callers should not need to construct
`SimulationStore` directly just to load analysis views.

For Matplotlib PSF, PSF-core, and metric-field plots from loaded analysis
simulations, use `ao_predict.plotting`. Plotting helpers live in that submodule
rather than the package root.

## Model Training

Use `train_model(TrainModelRequest(...))` to fit the AO Predict dense-regression
family. Training owns data validation, whole-simulation partitioning,
standardization, deterministic model construction, optimization, exact
continuation, and publication of a validated model package.

```python
import numpy as np

from ao_predict import (
    TrainModelRequest,
    model_training_data_from_rows,
    train_model,
)

features = np.asarray(
    [[0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5]],
    dtype=np.float32,
)
targets = np.asarray([[0.8], [1.2], [1.7], [2.1]], dtype=np.float32)
data = model_training_data_from_rows(
    features,
    targets,
    feature_names=("seeing", "airmass"),
    target_names=("fwhm",),
)

result = train_model(
    TrainModelRequest(
        model_path="models/example",
        training_data=data,
        validation_count=1,
        split_seed=17,
        hidden_widths=(32, 32),
        batch_size=64,
        training_seed=23,
    )
)
```

Exactly one validation source is required: `validation_data`,
`validation_count`, or `validation_fraction`. Automatic partitioning withholds
complete simulations. For explicit partitions, AO Predict checks the ordered
feature and target schema but trusts the caller to prevent scientific leakage
between the partitions.

The public data configuration holds one NumPy array per feature and target.
Targets have shape `(simulations,)` or
`(simulations, examples_per_simulation)` and must all share a shape. A feature
may have that complete shape or the compact `(simulations,)` shape; compact
features are expanded only while assembling a batch. Caller arrays are borrowed
without mutation until AO Predict has made its owned standardized `float32`
state, so callers must not mutate them while `train_model()` is active.

`model_path` is a path stem, not a directory. For `models/example`, training
owns these companions:

- `models/example.model.zip`: independently loadable prediction-model package
- `models/example.training.log`: output-only, append-only human record
- `models/example.recovery.pt`: transient exact-continuation state, removed
  after successful package publication and log finalization

A compatible recovery checkpoint is continued automatically. Set
`overwrite=True` to remove the complete derived output set and start again.
Incompatible recovery raises `TrainingRecoveryMismatchError`; invalid coupled
inputs raise `ModelTrainingValidationError` with all collected messages in
`issues`.

Training defaults to CPU. `device` accepts an explicit available `cpu`, `cuda`,
or `mps` device and never silently falls back. `cpu_threads` is optional and,
when set for CPU training, changes PyTorch's process-wide thread count. The
validation execution batch defaults to `2 * batch_size` and affects memory and
throughput, not metric semantics.

See the [Training API reference](reference/training.md) for the complete request,
result, and data contracts.

## Model Prediction And Evaluation

Use `load_model_predictor()` to load the validated `.model.zip` package
published by `train_model()`. The loader accepts either the model stem or the
exact package path; it resolves the path from the suffix alone and never picks
between candidates according to what exists.

```python
import numpy as np

from ao_predict import load_model_predictor

predictor = load_model_predictor(
    "models/example",
    device="cpu",
    batch_size=16_384,
)

features = np.asarray(
    [[0.75, 1.1], [1.25, 1.8]],
    dtype=np.float32,
)
predictions = predictor.predict(features)

targets = np.asarray([[0.95], [1.45]], dtype=np.float32)
evaluation = predictor.evaluate(features, targets)
print(evaluation.relative_rmse)
```

Direct feature input has shape `(examples, features)` and direct predictions
have shape `(examples, targets)`. Model metadata fixes the input and output
order, exposed through `feature_names` and `target_names`. Prediction always
returns a physical-unit `float32` NumPy array.

Named input avoids assembling a complete feature matrix and can keep values
shared at simulation scope:

```python
predictions = predictor.predict(
    {
        "seeing": np.asarray([0.6, 0.8], dtype=np.float32),
        "airmass": np.asarray(
            [[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]],
            dtype=np.float32,
        ),
    }
)
assert predictions.shape == (2, 3, 1)
```

Each named feature is either `(simulations,)` or
`(simulations, related_examples)`. Rank-one values are shared over the related
axis while AO Predict gathers each bounded batch; they do not need to be
repeated by the caller. When every feature is rank one, prediction returns
`(simulations, targets)`. Mapping insertion order is irrelevant, but names must
match the package exactly.

`predict_one()` accepts one positional feature vector or an exact-name mapping
of real scalars and returns `(targets,)`. `predict()` and `evaluate()` accept a
positive per-call `batch_size`; `None` uses the predictor default. A loaded
predictor also exposes its normalized `model_path`, exact `model_package_path`,
resolved `device`, names, and units as read-only properties.

Evaluation accepts direct arrays or exact-name mappings independently for
features and targets. Targets must provide every resolved example value, must
be finite and strictly positive, and are never broadcast. The immutable result
reports dimensionless ratios:

- `relative_mse`: mean squared relative residual pooled over all examples and
  targets
- `relative_rmse`: square root of that pooled value
- `target_relative_rmse`: complete-population relative RMSE by target name

These values are not percentages. Training's percent-valued validation Error
is `100 * relative_rmse` for the same model and population, within normal
floating-point tolerance.

Prediction defaults to CPU and supports explicit available PyTorch `cpu`,
`cuda`, `cuda:<index>`, and `mps` device names without automatic fallback.
Optional `cpu_threads` applies only to CPU and changes PyTorch's process-wide
thread count. Batch size bounds conversion, standardization, device transfer,
model execution, and physical reconstruction; AO Predict allocates the complete
output but does not assemble or repeat a complete named input matrix.
Numerically equivalent CPU and accelerator predictions are expected to agree
within `rtol=1e-5` and `atol=1e-6`; they are not required to be bitwise
identical.

See the [Prediction API reference](reference/prediction.md) for the exact
public signatures and result fields.

## Lifecycle Functions

### `init_dataset(request: InitDatasetRequest) -> int`
Initialize an HDF5 simulation dataset from code-provided config.

Responsibilities:
- create and validate simulation payload (`/simulation`)
- create and validate setup payload (`/setup`)
- complete and validate options payload (`/options`)
- allocate status/meta/stats (and optional psfs) datasets

Simulation payload note:
- ao-predict assembles the core `/simulation` fields: `name`, `version`, `extra_stat_names`, and `ngs_mag_standard`.
- Simulations expose the extra-stat registry through the `Simulation.extra_stat_names` property.
- `Simulation.ngs_mag_standard` declares the photometric standard of `/options/ngs_mag`; `BaseSimulation` supplies `R` as the default and subclasses override it when needed.
- The simulation implementation completes that base payload with simulation-specific persisted fields.
- New datasets require the complete current payload. Older datasets without `ngs_mag_standard` load with the historical `R` default and are not rewritten; see the [persisted-contract compatibility lifecycle](architecture.md#persisted-contract-compatibility).
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

Code-driven initialization may include independently optional
`sci_dx_arcsec` and `sci_dy_arcsec` matrices with shape `[N, M]`, where `M`
is the invariant science-point count in `/setup`. Retained matrices are finite
`float32`; an absent or all-zero axis means zero offset and is not persisted.
During execution, `SimulationContext.setup` remains invariant and the
effective polar field is available through `resolved_sci_r_arcsec` and
`resolved_sci_theta_deg`.

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
