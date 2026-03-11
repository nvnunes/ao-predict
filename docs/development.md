# Development Setup

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

## Code Organization Standard

Keep module/class structure consistent:
- constants first
- data structures next (dataclasses/types/enums)
- properties near the top (after constants and class definitions)
- helpers before public entrypoints
- use section comments for major blocks
- if a module has a strong lifecycle or execution flow, prefer ordering sections and methods by that lifecycle
