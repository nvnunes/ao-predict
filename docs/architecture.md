# Architecture

This document is the source of truth for `ao-predict` package structure, public API
boundaries, persisted-contract ownership, and simulation lifecycle.

## Shared Guidance

This repo adopts the shared guidance in:
- `astro-agents/guidance/agent-surface.md`
- `astro-agents/guidance/public-python-projects.md`
- `astro-agents/guidance/python-development.md`

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

## Extension Points And Hooks

Subclass hooks should prepare simulation-specific inputs and runtime state.
They should not redefine core persisted contracts.

For docstring expectations on class-contract hooks and published-reference
docstrings, follow the shared Python-development guidance.
