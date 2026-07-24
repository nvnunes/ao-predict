# Architecture

This document is the source of truth for `ao-predict` package structure, public API
boundaries, persisted-contract ownership, and simulation lifecycle.

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
- Future data-preparation or model-training modules should keep their own
  dedicated boundaries instead of growing out of simulation or persistence
  modules.

Core concerns stay in core modules. Simulation-specific behavior stays in
subclasses or feature modules.

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
