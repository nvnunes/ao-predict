# AGENTS.md

## Scope
- Documentation surface profile: public-python.

## Source Of Truth Docs
- Follow [`README.md`](README.md) for the repo's public summary and starting docs.
- Follow `docs/architecture.md` for repo structure, public API boundaries, persisted-contract ownership and simulation lifecycle rules.
- Follow `docs/testing.md` for canonical verification commands and completion expectations.
- Follow `docs/development.md` for bootstrap, local environment setup, and git hook activation.

## Shared Validation
- Use `$agent-surface-review` for shared agent-surface review.
- Use `$documentation-surface-review` for documentation-surface review with the `public-python` profile.
- Use `$code-quality-review` for source-code quality review.

## Skill Requirements
- For Python code, use `$python-code-writing`.
- For project docs such as `docs/architecture.md`, `docs/testing.md`, `docs/development.md`, and similar long-lived project documents, use `$project-docs-writing`.
- For `README.md`, use `$readme-writing`.
- For plan documents or phased execution docs when they are created or revised, use `$plan-writing`.

## Working Rules
- For package structure, public API boundaries, persisted contracts, and lifecycle-sensitive changes, consult `docs/architecture.md` before editing.
- Before concluding substantial work, satisfy the verification expectations in `docs/testing.md`.
- Use the local `./.conda` environment and the workflow in `docs/development.md` for Python commands, test runs, and docs builds unless a task explicitly requires something else.
