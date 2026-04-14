# Testing

This document is the source of truth for local verification commands and
completion expectations in `ao-predict`.

## Shared Validation

Use the shared base testing guidance in
`astro-agents/validation/base-testing.md`.

## Repo-Local Verification

Use the repo-local verification commands and completion expectations below.

## Environment

Use the local `./.conda` environment for Python commands, test runs, and docs
builds unless a task explicitly requires something else.

For fresh clones, use:

```bash
./scripts/bootstrap.sh
```

That script creates the local environment when needed, installs the package
with `dev` and `docs` extras, configures the git hooks path, runs the test
suite, and builds the docs in strict mode.

## Canonical Verification Commands

Refresh the local editable install when needed with:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

Run the Python test suite with:

```bash
./.conda/bin/python -m pytest -q
```

Build the docs with:

```bash
./.conda/bin/mkdocs build --strict
```

## Completion Expectations

Run the full test suite before concluding substantial refactors.

Always finish with the full test suite for changes that affect:

- persisted schema or validation behavior
- simulation lifecycle or runner behavior
- public API exports
- CLI behavior
- docs examples that describe executable workflows

Run the strict docs build for changes that affect:

- `README.md`
- `docs/*`
- examples or API/docs snippets
- package exports or docstrings that feed generated reference pages

Targeted tests are acceptable during iteration, but final verification should
match the changed surface area.

## Git Hook Behavior

The repo includes a versioned pre-commit hook at `.githooks/pre-commit`.

When active, it runs:

- `pytest -q`
- `mkdocs build --strict`

If the hooks path is not active in your clone, set it with:

```bash
git config core.hooksPath .githooks
```
