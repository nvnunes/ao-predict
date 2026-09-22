# Development Setup

This document covers local setup and bootstrap. For repo structure and
ownership rules, use `docs/architecture.md`. For canonical verification
commands and completion expectations, use `docs/testing.md`.

## Shared Skills

For Python code changes, use `$python-code-writing` alongside this project's
local environment and workflow rules.

Repo-local bootstrap commands, environment setup, toolchain choices, and hook
behavior in this document remain the source of truth for this repo.

## Bootstrap

Use the bootstrap script to configure a fresh clone:

```bash
AO_PREDICT_HYBRID_SOURCE=../hybrid-ao-psf \
./scripts/bootstrap.sh
```

`hybrid-ao-psf>=0.1.0` and `ngs-photometry>=0.1.0` are required distribution
dependencies. Bootstrap installs NGS Photometry from its GitHub repository
before AO Predict and its extras. Set `AO_PREDICT_NGS_PHOTOMETRY_SOURCE` to a
sibling checkout for editable development, or to a Git URL at a verified
revision. Set `AO_PREDICT_HYBRID_SOURCE` to a sibling checkout or Git source
when Hybrid AO PSF is unavailable from the configured package index. Plain
`./scripts/bootstrap.sh` is sufficient when Hybrid AO PSF is available from
that index.

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

When developing against sibling dependency checkouts, refresh their editable
installations separately after upstream changes:

```bash
./.conda/bin/python -m pip install -e ../hybrid-ao-psf
./.conda/bin/python -m pip install -e ../ngs-photometry
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
