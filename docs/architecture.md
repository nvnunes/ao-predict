# Architecture

This document is the source of truth for `ao-predict` package structure, public API
boundaries, persisted-contract ownership, and simulation and model lifecycles.

## Shared Validation And Skills

This project uses runtime-discovered `astro-agents` skills for shared review and authoring support:

- `$agent-surface-review`
- `$documentation-surface-review` with the `public-python` profile
- `$code-quality-review`
- `$python-code-writing`

Repo-local package boundaries, persisted contracts, lifecycle rules, code
organization priorities, and exceptions in this document remain the source of
truth for this repo.

## Package Surface

`ao_predict` is the deliberate public Python API boundary.

- Re-export only supported user-facing types and entrypoints from the package
  root.
- Keep lower-level orchestration, persistence helpers, and internal wiring in
  their natural modules.
- Keep the CLI as a thin wrapper over the Python API.

## Module Boundaries

Keep one obvious owner for each major concern:

- Simulation execution belongs under `simulation/*`.
- Persistence and storage concerns belong under `persistence/*`.
- Analysis read models and load composition belong under `analysis/*`.
- Generic analysis plotting belongs in `plotting`, including PSF views,
  centered PSF-core views, and interpolated metric-field views over persisted
  science coordinates.
- Model-training data, lifecycle, and package publication belong under
  `training/*` rather than growing out of simulation or persistence modules.
- Deployable model loading, prediction, and aggregate evaluation belong under
  `prediction/*`.
- Dense-model construction, device selection, and model-package validation
  shared by training and prediction have one private package-level owner.

Core concerns stay in core modules. Simulation-specific behavior stays in
subclasses or feature modules.

## Cross-Lifecycle API Consistency

Simulation and model-training APIs should express similar problems with the
same project vocabulary and ownership pattern:

- use passive immutable public configuration objects and prepare private
  validated state at the operation boundary
- use one flat lifecycle request, one public operation, and one immutable
  result rather than exposing orchestration internals
- collect coupled input problems on a focused `ValueError` subclass with a
  public list of messages
- keep persisted-contract helpers and mutable execution state private
- choose names and field order that follow the lifecycle and match established
  AO Predict terms

New model-lifecycle behavior should reuse these patterns unless its semantics
require a deliberate difference. It should not introduce a second naming
system or a parallel abstraction for a problem already solved by another AO
Predict API.

## Model Training Lifecycle

`ao_predict.training` owns the dense-regression training lifecycle. Its public
boundary consists of feature and target configurations,
`ModelTrainingDataConfig`, `TrainModelRequest`, `train_model()`, the training
result records, and focused validation and recovery-mismatch errors. Model
construction, standardized prepared state, partition membership, optimizer and
scheduler machinery, recovery mappings, locks, and package persistence remain
private.

Training follows this ownership sequence:

1. Validate the complete request and borrow caller NumPy arrays without
   mutation or full-matrix duplication.
2. Resolve explicit validation or split complete simulations from one pool.
3. Fit population standardizers on training observations and create owned
   standardized `float32` arrays.
4. Construct and initialize the supported dense model from a private random
   stream, then fit and validate in bounded batches.
5. Save exact recovery after each completed validation boundary.
6. Publish and independently validate the best model package, finalize the
   human log, and only then remove recovery state.

The first axis of every training values array is the simulation axis. Targets
have either one value per simulation or one common second example axis. Each
feature may match the target shape or remain compact with one value per
simulation. Automatic partitioning operates only on the simulation axis;
training shuffles the resulting scalar examples.

### Training Clocks

One aligned example position supplies one feature vector and one target vector
to the model. A simulation may contribute one example or multiple related
examples. Partitioning keeps complete simulations together, while fitting
shuffles and batches the resolved example positions.

For `N` training examples and training batch size `B`, one epoch contains
`ceil(N / B)` optimizer updates. A smaller final batch uses every remaining
example and still produces one update. AO Predict does not accumulate
gradients across batches.

The lifecycle tracks several clocks because they answer different questions:

- `optimizer_updates` counts parameter updates.
- `training_examples_seen` counts examples consumed by training batches.
- `training_epochs` is `training_examples_seen / N` and may therefore be
  fractional at a validation boundary.
- `validation_checks` counts complete evaluations of the validation
  partition.

Warmup and the minimum-training eligibility gate are configured in epochs and
resolved to optimizer updates using `ceil(N / B)`. Validation cadence is
configured with `validation_check_epochs` and resolved to an example-exposure
threshold of `ceil(validation_check_epochs * N)`. AO Predict runs a validation
check after the batch that first reaches or exceeds that threshold, then resets
the validation-exposure counter. A validation check may therefore occur within
an epoch, and the final batch before a check may carry exposure slightly past
the threshold.

### Objective And Complete Validation

AO Predict fits the supported dense model with physical relative MSE. For each
strictly positive physical target, the relative residual is
`(prediction - target) / target`. The objective is the mean squared relative
residual over all evaluated examples and target elements.

The lifecycle constructs PyTorch Adam internally. Callers select the base
learning rate and coupled weight decay; AO Predict fixes `betas=(0.9, 0.999)`,
`eps=1e-8`, `amsgrad=False`, and `maximize=False`, enables `foreach` only on
CUDA, and disables fused execution. These choices belong to the supported
training lifecycle rather than to a caller-supplied optimizer plug-in.

Training batches calculate the objective needed for their optimizer update.
At a validation check, AO Predict predicts the complete validation partition
in bounded execution batches, accumulates squared relative residuals over the
complete population, and divides once. It does not average independently
calculated batch objectives or errors. Validation Error is the square root of
that complete validation objective expressed as a percentage.

Each `TrainingValidationRecord` describes one validation boundary. Its
training objective covers the training examples consumed since the preceding
check, or since fitting began for the first record. Its validation objective
and Error cover the complete validation partition. Learning-rate fields record
the value before and after scheduler processing at that boundary.

### Scheduling And Stopping

The learning-rate scheduler and early stopping use independent monitors,
threshold units, eligibility gates, and improvement references:

- Warmup increases the learning rate linearly from
  `warmup_start_fraction * base_learning_rate` to `base_learning_rate`. The
  scheduler neither acts nor accumulates patience before warmup is complete.
- The scheduler becomes eligible after warmup. It monitors validation
  objective and requires the configured relative decrease from its best
  eligible objective. `scheduler_patience_checks` names the consecutive
  unsuccessful check on which the learning rate is reduced.
- Early stopping becomes eligible only after minimum training is complete and
  the scheduler has reduced the learning rate at least once. Its first eligible
  check establishes a validation-Error reference. It then requires the
  configured absolute decrease in percentage points from the last qualifying
  reference; smaller decreases may accumulate toward that threshold.
  `early_stopping_patience_checks` names the consecutive unsuccessful check on
  which training terminates.

Every scheduled validation check, including checks during warmup or before
early stopping becomes eligible, counts toward `maximum_validation_checks`.
When early stopping and the maximum bound act on the same check, the completed
run reports early stopping as its termination reason.

A validation check applies lifecycle actions in this order:

1. Calculate and record the complete validation objective and Error.
2. Replace the best-model state when the validation objective is the lowest
   observed in the run.
3. Apply eligible scheduler processing to the validation objective.
4. Apply eligible early-stopping processing to validation Error.
5. Apply the maximum-validation-check bound.
6. Retain the complete updated recovery state.

### Model Selection And Exact Continuation

The published model is the state from the lowest validation objective, not
necessarily the final training state. Its validation Error is retained
separately because the lowest objective need not imply a separately selected
lowest Error.

Recovery state serves a different purpose from best-model selection. It
retains the current and best model states, optimizer and scheduler state,
batch order and cursor, private random streams, preprocessing, progress,
history, runtime identity, and publication state needed to continue the same
logical run. Exact continuation appends to the existing validation history and
keeps the original run identity. Terminal recovery performs no further fitting
and exists only long enough to finish model-package publication and log
finalization safely.

The [Python API guide](api.md#model-training) owns the executable usage path,
and the [training API reference](reference/training.md) owns the exact public
fields and defaults. This architecture document owns how those controls
interact across the lifecycle.

### Training Persisted Contracts

One caller-selected `model_path` owns a tightly coupled set of sibling files:

- `<model_path>.model.zip` is the immutable deployable package. It contains
  exactly `manifest.json`, `metadata.json`, and weights-only `weights.pt`.
- `<model_path>.training.log` is an output-only, append-only human record. No
  execution or loading path parses it.
- `<model_path>.recovery.pt` is transient weights-only exact-continuation
  state. It contains identity, standardized parameters, lifecycle state, and
  private random-stream state, but no source feature or target arrays.

Package and recovery formats have independent `kind` and integer `version`
contracts. Model packages are portable across supported inference runtimes;
their producer version is provenance. Exact recovery additionally requires an
exact training-runtime fingerprint and the original caller data to reproduce
the retained schema, dtype, shape, and content checksums.

Publication uses temporary siblings and atomic replacement. A private
exclusive lock covers inspection, fitting, logging, recovery replacement, and
publication for one `model_path`. `overwrite=True` authorizes replacement of
the stable derived output set but never bypasses an active lock.

## Model Prediction And Evaluation Lifecycle

`ao_predict.prediction` owns the deployed dense-model runtime. Its public
boundary consists of `load_model_predictor()`, `ModelPredictor`, and
`ModelEvaluationResult`. The loader validates one versioned model package,
reconstructs the concrete model family, moves the model and fitted
standardization state to the selected device, and returns a runtime whose
public state is read-only. Raw model state, weights, metadata mappings, and
scaler tensors remain private.

Prediction follows this sequence:

1. Resolve a model stem or exact `.model.zip` path syntactically.
2. Validate the package manifest, member integrity, metadata, weights, and
   reconstructed dense-model state.
3. Resolve the explicit runtime device and optional process-wide CPU thread
   setting without automatic fallback.
4. Validate and bind caller feature values to the package's ordered feature
   contract.
5. Gather, convert, standardize, execute, and reconstruct physical values in
   bounded batches under model evaluation and inference modes.
6. Restore the caller-visible example shape and return a plain `float32` NumPy
   array.

Training and prediction deliberately have different public data forms.
Training defines and persists a feature and target schema, so its public data
configuration pairs each array with a definition. Prediction consumes values
against that already-fixed package schema, so callers may provide either one
positional matrix or an exact-name mapping without repeating definitions or
units. Internally, both lifecycles preserve simulation-major order, treat the
last partial batch normally, and gather simulation-level features without
permanently repeating them.

Named prediction features may be rank one at simulation scope or rank two over
a common related-example axis. The runtime creates only the current dense
feature batch: it does not build a complete combined input matrix or repeat a
complete simulation-level feature. Direct inputs and complete outputs remain
caller-shaped. Prediction accepts an empty example population and returns the
corresponding empty output; evaluation requires at least one example.

Evaluation uses the same prediction pipeline but accumulates squared physical
relative residuals instead of retaining predictions. Targets must match the
resolved example shape exactly, are never broadcast, and must be strictly
positive. The runtime sums batch contributions and divides once over the
complete population. Overall relative MSE pools all examples and targets;
overall relative RMSE is its square root; per-target relative RMSE uses each
target's complete example population. These values are dimensionless ratios,
while the training lifecycle's validation Error expresses the same pooled RMSE
as a percentage.

Model packages are device-independent. CPU and accelerator execution must
agree within the public API guide's documented floating-point tolerance, not
bitwise identity. Runtime device and batch selection do not alter package
identity.

The `.model.zip` package is the only deployed-model input. Prediction does not
read the training log or recovery checkpoint and does not support legacy
packages, caller-provided model factories, alternate model families, custom
statistics, study orchestration, or execution benchmarking.

The [Python API guide](api.md#model-prediction-and-evaluation) owns executable
usage, and the [prediction API reference](reference/prediction.md) owns the
exact public signatures and result fields.

## Persisted Contract Ownership

Treat persisted simulation payloads, setup data, options data, and analysis
load inputs as explicit contracts.

- Validate early and return actionable errors.
- Avoid silent coercions or hidden behavior changes.
- Validate coupled multi-field inputs as one logical family rather than as
  unrelated independent keys.
- Define persisted keys, required-key collections, and stable field maps as
  named constants in the narrowest shared module that owns the contract.

Keep one clear owner per rule:

- schema and key definitions in schema or contract modules
- persisted payload validation in core validation modules
- payload preparation and defaulting in prepare or build paths
- typed state binding in load or bind paths

Avoid split ownership where builders, validators, and subclasses all partially
enforce the same persisted rule.

## Persisted Contract Compatibility

Newly prepared payloads must satisfy the current persisted contract. Dataset
creation validates that contract directly and does not apply compatibility
upgrades.

Payload loading first validates the current contract. When validation fails,
AO Predict may apply a recognized legacy upgrade to an in-memory copy and then
validates the current contract again. An upgrade never rewrites the persisted
dataset, and consumers receive only the validated current-contract payload.

Keep each legacy upgrade explicit, narrow, and idempotent. Do not use an
upgrade to bypass unrelated validation failures.

## Simulation Lifecycle

Keep preparation, validation, and binding clearly separated:

- `prepare_*`: build or complete persisted payloads
- core validation modules: enforce persisted schema and contract
- `load_*`: deserialize and bind typed state

Core-owned fields and behaviors should be validated and computed in core
modules. Subclass hooks should handle only simulation-specific behavior.

Builders may normalize inputs and apply defaults, but final
persisted-contract enforcement belongs in core validation modules.

Avoid validating by temporarily mutating bound instance state and then
restoring it.

If a module has a strong lifecycle or execution flow, prefer method order that
follows that lifecycle.

## Per-Simulation Science Coordinates

The persisted `/setup/sci_r_arcsec` and `/setup/sci_theta_deg` vectors define
the invariant base science grid. Code-driven initialization may additionally
provide `/options/sci_dx_arcsec` and `/options/sci_dy_arcsec` matrices with
shape `[N, M]`, where `N` is the simulation count and `M` is the base-grid
point count.

- Each Cartesian offset axis is independently optional and defaults to zero
  when absent.
- Persisted offset matrices use finite `float32` values. An all-zero axis is
  normalized away instead of being stored or loaded.
- Execution resolves one options row against the base grid and records the
  effective polar coordinates in
  `SimulationContext.resolved_sci_r_arcsec` and
  `SimulationContext.resolved_sci_theta_deg`.
- `SimulationContext.setup`, the bound setup, and persisted `/setup` remain
  invariant.
- TIPTOP and Hybrid execution consume the effective runtime coordinates.
- Analysis and plotting continue to expose the invariant setup grid; adopting
  effective coordinates in those surfaces is a separate change.

The offset matrices are an `OptionsConfig` API capability. The CLI does not
provide a dedicated input surface for them.

## Extension Points And Hooks

Subclass hooks should prepare simulation-specific inputs and runtime state.
They should not redefine core persisted contracts.

For docstring expectations on class-contract hooks and published-reference
docstrings, follow the shared Python-development guidance.
