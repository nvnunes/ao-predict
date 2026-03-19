# CLI Documentation

This document describes the `ao-predict` command-line interface for simulation dataset lifecycle management.

## Command Structure
```bash
ao-predict [--version] simulate <subcommand> [options]
```

`ao-predict --version` prints the package version and exits.

Subcommands:
- `init`
- `run`
- `retry`
- `reset`
- `check`

## `simulate init`
Initialize a dataset file from a YAML configuration.

```bash
ao-predict simulate init <config_yaml> [--dataset <path>] [--overwrite] [--save-psfs]
```

Arguments:
- `config_yaml`: path to YAML simulation config.

Options:
- `--dataset`: explicit output HDF5 path.
- `--overwrite`: overwrite existing dataset file.
- `--save-psfs`: allocate `/psfs/data` and persist PSF cubes.

Behavior:
- If `--dataset` is omitted, dataset path defaults to `<config-dir>/sims/<config-stem>.h5`. The `sims/` folder is created automatically if needed.
- Simulation metadata is written under `/simulation`.
- Invariant setup values are written under `/setup`.
- Per-simulation options are written under `/options`.
- Initial status is `SimulationState.PENDING` (`0`) for all simulations.

## `simulate run`
Run all pending simulations.

```bash
ao-predict simulate run <dataset> [--verbose] [--sims 2,5,8]
```

Behavior:
- Validates schema.
- Loads simulation and setup from dataset.
- Runs only simulations where `/status/state == SimulationState.PENDING` (`0`).
- With `--sims`, runs only the selected simulation numbers (1-based) that are pending.
- With `--verbose`, prints failure messages for failed simulations.

Output:
- `Run summary: attempted=<N> succeeded=<S> failed=<F>`

## `simulate retry`
Retry failed simulations only.

```bash
ao-predict simulate retry <dataset> [--verbose] [--sims 2,5,8]
```

Behavior:
- Validates schema.
- Loads simulation and setup from dataset.
- Runs only simulations where `/status/state == SimulationState.FAILED` (`2`).
- With `--sims`, retries only the selected simulation numbers (1-based) that are failed.
- Keeps successful simulations unchanged.
- With `--verbose`, prints failure messages for failed simulations.

Output:
- `Retry summary: attempted=<N> succeeded=<S> failed=<F>`

## `simulate check`
Validate schema and completion status.

```bash
ao-predict simulate check <dataset>
```

Behavior:
- Runs dataset schema validation.
- Reports pending and failed counts.

Exit code:
- `0` when dataset is valid and all simulations are successful.
- `1` when schema errors or unfinished/failed simulations exist.

Output examples:
- Pass:
  - `Dataset check PASSED: ...`
  - `All simulations completed successfully (N=<num_sims>).`
- Fail:
  - `Dataset check FAILED: ...`
  - issue list lines prefixed with `-`

## `simulate reset`
Reset all simulations to pending state.

```bash
ao-predict simulate reset <dataset> [--sims 2,5,8]
```

Behavior:
- Validates schema.
- With no `--sims`, sets every `/status/state` value to `SimulationState.PENDING` (`0`).
- With `--sims`, resets only the selected simulation numbers (1-based).
- Keeps existing `/stats`, `/meta`, and `/psfs` data in place; reruns overwrite results as simulations complete.

Output:
- `Reset summary: changed=<C>`

## YAML Configuration Reference

Top-level sections:
- `simulation`
- `setup`
- `options`

Key casing:
- YAML and CSV keys are accepted in any case by the CLI.
- CLI normalizes all keys to lowercase before calling the API.
- Use lowercase keys in examples/specs.

### `simulation`
Required:
- `name`: simulation class identifier.

TIPTOP usage:
- `name: Tiptop` (short form) or `ao_predict.simulation.tiptop:TiptopSimulation`.
- `config_path`: path to source INI file.

Any extra keys in `simulation` are passed through to the simulation implementation.

### `setup`
Core required key:
- `ee_apertures_mas`

Most simulation-specific setup values are resolved by the simulation implementation (for TIPTOP, usually from INI).
For `TiptopSimulation`, `setup.ngs_mag_zeropoint` is also required.

### `options`
Three supported inputs:

1. Broadcast defaults (single values):
```yaml
options:
  wavelength_um: 1.65
  zenith_angle_deg: 20
```

2. Inline table:
```yaml
options:
  table:
    columns: [wavelength_um, zenith_angle_deg, atm_profile_id, r0_m, ngs1_r_arcsec, ngs1_theta_deg, ngs1_mag]
    rows:
      - [1.65, 20, 0, 0.16, 10.0, 0.0, 14.0]
      - [1.65, 25, 0, 0.14, 12.0, 30.0, 15.0]
```

3. CSV table:
```yaml
options:
  table_path: path/to/options.csv
```

Rule:
- `table` and `table_path` are mutually exclusive.
- CSV column names are lowercased by CLI parsing.
- The persisted `/options` payload always contains the NGS triplet.
- If you provide any of `ngs*_r_arcsec`, `ngs*_theta_deg`, or `ngs*_mag`, provide the full triplet.
- Unused star slots may be represented with `NaN` after normalization, but each slot must be either all finite or all `NaN` across radius, angle, and magnitude.
- If you omit the NGS triplet entirely, the selected simulation must supply it during options preparation.

Precedence:
- table values first
- broadcast values fill missing values
- simulation completion logic fills remaining required option keys from simulation defaults

Atmospheric input note:
- `r0_m` is the canonical persisted option in `/options`.
- `seeing_arcsec` is accepted as an input alias (table/broadcast), converted to `r0_m` using `setup.atm_wavelength_um`, and is not persisted.
- If both `r0_m` and `seeing_arcsec` are provided for one simulation, they must be consistent.
- In `setup.atm_profiles`, `seeing_arcsec` is also accepted per profile, normalized to `r0_m`, and not persisted.

## Dataset Layout
Top-level groups:
- `/simulation`
- `/setup`
- `/options`
- `/status`
- `/meta`
- `/stats`
- optional `/psfs`

Stats layout:
- `/stats/sr`: core `[N, M]`
- `/stats/ee`: core `[N, M, A]`
- `/stats/fwhm_mas`: core `[N, M]`, selected by `/setup/fwhm_summary`
- Additional `/stats/*` datasets may appear when declared by the simulation in `/simulation/extra_stat_names`; each extra stat dataset is `[N, M]`.

Setup-level stats selectors:
- `/setup/sr_method`: dataset-level Strehl selector, `pixel_fit` or `pixel_max`
- `/setup/fwhm_summary`: dataset-level contour-summary selector, `geom`, `mean`, `max`, or `min`

Core state dataset:
- `/status/state`: `uint8[N]`

State values:
- `0`: pending
- `1`: success
- `2`: failed

## Example Files
- API-driven example script: `examples/simulate_tiptop_api.py`
- CLI YAML config: `examples/simulate_tiptop_cli_example1.yaml`
- CLI YAML config with CSV table: `examples/simulate_tiptop_cli_example2.yaml`
- CLI CSV options table: `examples/simulate_tiptop_cli_example2.csv`
- CLI shell script: `examples/simulate_tiptop_cli.sh` (`1` by default, pass `2` for the CSV-table example)
- Sample TIPTOP INI: `examples/sample_tiptop.ini`
