# Development Setup

This document covers local setup and bootstrap. For repo structure and
ownership rules, use `docs/architecture.md`. For canonical verification
commands and completion expectations, use `docs/testing.md`.

## Shared Guidance

This repo adopts the shared guidance in
`astro-agents/guidance/public-python-projects.md` and
`astro-agents/guidance/python-development.md`.

Repo-local bootstrap commands, environment setup, toolchain choices, and hook
behavior in this document remain the source of truth for this repo.

## Bootstrap

Use the bootstrap script to configure a fresh clone:

```bash
./scripts/bootstrap.sh
```

The script will:
- create `.conda` if missing
- install package extras: `dev` and `docs`
- set `git config core.hooksPath .githooks`
- run `pytest -q`
- run `mkdocs build --strict`

## Daily Commands

After bootstrap, prefer commands from the local environment instead of bare
`python`, `pip`, or `mkdocs` invocations:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
./.conda/bin/python -m pytest -q
./.conda/bin/mkdocs build --strict
./.conda/bin/mkdocs serve
```

Run the CLI from the same environment:

```bash
./.conda/bin/ao-predict --version
```

## Docs Toolchain Note

The repo currently treats the MkDocs 1.x stack as the supported docs toolchain.
The dependency bound in `pyproject.toml` intentionally keeps `mkdocs` below
`2.0` until the repo chooses a deliberate migration path.

## Pre-commit Hook

The repo includes a versioned hook at:

- `.githooks/pre-commit`

On each commit, it runs:
- tests: `pytest -q`
- docs build: `mkdocs build --strict`

If hooks are not active in your clone, run:

```bash
git config core.hooksPath .githooks
```
