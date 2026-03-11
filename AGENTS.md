# AO-Predict Agent Brief

## Scope
- Maintain high code quality across simulation, persistence, data loading, training, and inference.
- Prefer stable contracts and incremental refactors over broad rewrites.
- Prefer removing stale abstractions over preserving weak indirection.

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
  - `prepare_*`: build/complete persisted payloads
  - core validation modules: enforce persisted schema/contract
  - `load_*`: deserialize and bind typed state
- Avoid validating by temporarily mutating bound instance state and then restoring it.
- Builders may normalize inputs and apply defaults, but final persisted-contract enforcement belongs in core validation modules.
- Subclass hooks should prepare simulation-specific inputs and runtime state, not redefine core persisted contracts.

## CLI/API Principles
- CLI is a thin wrapper over API.
- API should be clean, typed where useful, and minimally surprising.
- Public docs and examples should reflect the supported import path and public API surface, not internal module structure.

## Data & Model Discipline
- Use deterministic behavior for data splits, preprocessing, and training when requested.
- Keep feature/target schemas explicit and version-aware.
- Keep model bundle metadata tightly coupled to data/schema compatibility.

## Code Style
- Optimize for readability, homogeneity, and symmetry in naming/order.
- Keep module order consistent with this codebase style:
  - constants first
  - data structures next (dataclasses/types/enums)
  - properties near the top (after constants and class definitions)
  - helper primitives next
  - composed helpers next
  - public entrypoints last
- If a module has a strong lifecycle or execution flow, prefer method order that follows that lifecycle, even if it overrides the default ordering rule.
- When lifecycle order is the primary organizing principle, group helpers with the lifecycle stage they support instead of collecting all helpers in one block.
- separate blocks with clear section comments
- Core-owned fields and behaviors should be validated and computed in core modules. Subclass hooks should handle only simulation-specific behavior.
- Prefer explicit names that match actual behavior. Do not use “normalize” or similar language when a helper only validates or coerces.
- Prefer inlining helpers that do too little to justify abstraction. Small wrappers should survive only when they clarify ownership, lifecycle stage, or contract meaning.
- If two helpers/validators enforce the same invariant in the same flows, collapse them into one shared helper instead of layering wrappers.
- Prefer positional parameters for private methods unless keyword-only arguments provide clear safety or readability benefits. Required named parameters are acceptable for public APIs, but should be used sparingly in internal helpers.
- When normalizing values inside private methods, prefer rebinding the original variable name instead of introducing `_arr`-style aliases unless both forms need to coexist.
- Keep comments concise and technical; avoid historical/conversational comments.
- Move helpers out of large classes when they are no longer lifecycle-specific. Place them in the narrowest shared module that matches their domain instead of a generic dumping ground.
- Underscore-prefixed methods that serve as subclass hooks or lifecycle extension points are part of the class contract. Document them with full docstrings covering purpose, inputs, return value, mutation expectations, and error behavior, even if they are not public APIs.
- Push back before implementing changes that reduce clarity or consistency.

## Docstrings
- Keep docstrings aligned with the current code. If behavior or ownership moves, update the docstrings in the same change.
- Public functions and core validators should document the contract they enforce, not implementation history.

## Quality Bar
- Add/adjust tests with every behavior change.
- Keep docs and examples synchronized with code.
- Run full test suite before concluding substantial refactors.
- When a review uncovers a stale comment/docstring or duplicated contract check in touched code, clean it up in the same change rather than leaving follow-up debt.

## Review Lens
- Prefer concrete findings over speculative redesigns.
- Favor symmetry, contract ownership, lifecycle clarity, and removal of stale abstractions.
- If a hook or abstraction is no longer carrying its weight, prefer removing it to keeping a misleading no-op layer.
