# ao-predict
AO Predict: a framework for fast AO performance prediction

Current simulation support focuses on batched TIPTOP-style runs with resumable HDF5 persistence.

## Installation
```bash
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

For full local development setup (env, hooks, tests, docs):
```bash
./scripts/bootstrap.sh
```

## Quickstart: CLI
1. Initialize a dataset from the example YAML:
```bash
ao-predict simulate init examples/simulate_tiptop_cli_example1.yaml
```
2. Run pending simulations:
```bash
ao-predict simulate run examples/sims/simulate_tiptop_cli_example1.h5
```
3. Validate dataset completeness and schema:
```bash
ao-predict simulate check examples/sims/simulate_tiptop_cli_example1.h5
```
To retry failed simulations:
```bash
ao-predict simulate retry examples/sims/simulate_tiptop_cli_example1.h5
```
To reset all simulations, or selected simulation numbers, back to pending:
```bash
ao-predict simulate reset examples/sims/simulate_tiptop_cli_example1.h5 --sims 2,5
```

CLI key casing:
- YAML/CSV keys can be any case; CLI normalizes them to lowercase.

Full CLI documentation: [`docs/cli.md`](docs/cli.md)

## Quickstart: Python API
```python
from pathlib import Path

from ao_predict import (
    InitDatasetRequest,
    SetupConfig,
    SimulationConfig,
    SimulationState,
    TableOptionsConfig,
    check_dataset,
    init_dataset,
    reset_simulations,
    run_simulations_by_state,
)

request = InitDatasetRequest(
    dataset_path="examples/sims/demo.h5",
    simulation=SimulationConfig(
        name="Tiptop",
        base_path="examples",
        specific_fields={"config_path": "sample_tiptop.ini"},
    ),
    setup=SetupConfig(
        ee_apertures_mas=[50.0, 100.0],
        specific_fields={"ngs_mag_zeropoint": 3.0e10},
    ),
    options=TableOptionsConfig(
        broadcast={"zenith_angle_deg": 20.0},
        columns=["wavelength_um"],
        rows=[[1.654], [2.179]],
    ),
    overwrite=True,
    save_psfs=False,
)

num_sims = init_dataset(request)
dataset_path = Path(request.dataset_path)

summary = run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
status = check_dataset(dataset_path)
print(num_sims, summary, status.ok, status.issues)
```

To rerun simulations after a successful pass, call `reset_simulations(dataset_path)`
explicitly and then run pending simulations again.

Full API documentation: [`docs/api.md`](docs/api.md)

API key casing:
- Mapping keys are case-sensitive and must be lowercase.

## Documentation Site (MkDocs)
Install docs dependencies:
```bash
pip install -e ".[docs]"
```

Run local docs server:
```bash
mkdocs serve
```

Build docs in strict mode:
```bash
mkdocs build --strict
```

Pre-commit checks are versioned in `.githooks/pre-commit`.
If your clone is not using that hooks path, run:
```bash
git config core.hooksPath .githooks
```

## Working Examples
- API script: `examples/simulate_tiptop_api.py`
- CLI YAML config: `examples/simulate_tiptop_cli_example1.yaml`
- CLI YAML config with CSV table: `examples/simulate_tiptop_cli_example2.yaml`
- CLI CSV options table: `examples/simulate_tiptop_cli_example2.csv`
- CLI shell script: `examples/simulate_tiptop_cli.sh` (`1` by default, pass `2` for the CSV-table example)
- Sample TIPTOP INI: `examples/sample_tiptop.ini`

## Dataset State Semantics
- `0`: pending
- `1`: completed successfully
- `2`: failed

## Documentation Index
- CLI reference: [`docs/cli.md`](docs/cli.md)
- API reference: [`docs/api.md`](docs/api.md)
