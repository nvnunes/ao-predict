# AGENTS.md

## Prompt Routing
- Follow any higher-level workspace prompt-routing instructions when present.
- Repo-specific instructions in this file take precedence within this repository.

## Scope
- Maintain high code quality across simulation, persistence, data loading, training, and inference.
- Prefer stable contracts and incremental refactors over broad rewrites.
- Prefer removing stale abstractions over preserving weak indirection.
- Use the local `./.conda` environment for Python commands, test runs, and related tooling unless a task explicitly requires something else.

## Architecture
- Keep clear boundaries:
  - Simulation execution: `simulation/*`
  - Persistence/schema: `persistence/*`
  - Data preparation/loading: dedicated data modules
  - Training/inference/model bundles: dedicated model modules
- Core concerns stay in core modules; simulation/model-specific logic stays in subclasses or feature modules.
- Treat the package root (`ao_predict`) as a deliberate public API boundary. Re-export only supported user-facing types and entrypoints there; keep lower-level orchestration and persistence internals in their natural modules.
- Keep one obvious owner per contract:
  - schema/key definitions in schema/contract modules
  - persisted payload validation in core validation modules
  - payload preparation/defaulting in preparation/build modules
  - typed state binding in load/bind methods
- Avoid split ownership where builders, validators, and subclasses all partially enforce the same persisted rule.

## Contracts
- Treat persisted data schema and model/data specs as explicit contracts.
- Validate early with actionable errors.
- Avoid silent coercions or hidden behavior changes.
- Coupled multi-field inputs should be validated as one logical family, not as unrelated independent keys.
- Define persisted keys, required-key collections, and stable field maps as named constants in the narrowest shared module that owns the contract.
- Do not reconstruct contract key sets or duplicate string literals in multiple validators/helpers when a named constant should exist.

## Simulation Lifecycle
- Keep preparation, validation, and binding clearly separated:
  - `prepare_*`: build or complete persisted payloads
  - core validation modules: enforce persisted schema and contract
  - `load_*`: deserialize and bind typed state
- Avoid validating by temporarily mutating bound instance state and then restoring it.
- Builders may normalize inputs and apply defaults, but final persisted-contract enforcement belongs in core validation modules.
- Subclass hooks should prepare simulation-specific inputs and runtime state, not redefine core persisted contracts.
- If a module has a strong lifecycle or execution flow, prefer method order that follows that lifecycle.

## CLI And Public API
- CLI is a thin wrapper over API.
- API should be clean, typed where useful, and minimally surprising.
- Public docs and examples should reflect the supported import path and public API surface, not internal module structure.

## Data And Model Discipline
- Use deterministic behavior for data splits, preprocessing, and training when requested.
- Keep feature/target schemas explicit and version-aware.
- Keep model bundle metadata tightly coupled to data/schema compatibility.

## Repo-Specific Python Guidance
- Core-owned fields and behaviors should be validated and computed in core modules. Subclass hooks should handle only simulation-specific behavior.
- When lifecycle order is the primary organizing principle, it may override more generic module-order defaults.
- Underscore-prefixed methods that serve as subclass hooks or lifecycle extension points are part of the class contract. Document them with full docstrings covering purpose, inputs, return value, mutation expectations, and error behavior, even if they are not public APIs.

## Quality Bar
- Run the full test suite before concluding substantial refactors.

## Review Lens
- Favor contract ownership and lifecycle clarity in review.
