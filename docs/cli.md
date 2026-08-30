# CLI Documentation

This document describes the `ao-predict` command-line interface for simulation
dataset lifecycle management.

## Command Structure

```bash
ao-predict [--version] <command-family> <subcommand> [options]
```

`ao-predict --version` prints the package version and exits.

Command families:
- `simulate`: simulation dataset lifecycle management

Simulation subcommands:
- `init`
- `run`
- `retry`
- `resume`
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
ao-predict simulate run <dataset> [--verbose] [--sims 2,5,8] [--threads 4] [--chunks 10]
```

Behavior:
- Validates schema.
- Loads simulation and setup from dataset.
- Runs only simulations where `/status/state == SimulationState.PENDING` (`0`).
- With `--sims`, runs only the selected simulation numbers (1-based) that are pending.
- With `--threads`, uses joblib/loky worker processes. The default is serial execution.
- With `--chunks`, sets the number of simulations assigned to each worker chunk.
- With `--verbose`, prints failure messages for failed simulations.
- Worker processes return completed results; dataset writes remain in the parent process.

Output:
- `Run summary: attempted=<N> succeeded=<S> failed=<F>`

## `simulate retry`

Retry failed simulations only.

```bash
ao-predict simulate retry <dataset> [--verbose] [--sims 2,5,8] [--threads 4] [--chunks 10]
```

Behavior:
- Validates schema.
- Loads simulation and setup from dataset.
- Runs only simulations where `/status/state == SimulationState.FAILED` (`2`).
- With `--sims`, retries only the selected simulation numbers (1-based) that are failed.
- With `--threads`, uses joblib/loky worker processes. The default is serial execution.
- With `--chunks`, sets the number of simulations assigned to each worker chunk.
- Keeps successful simulations unchanged.
- With `--verbose`, prints failure messages for failed simulations.
- Worker processes return completed results; dataset writes remain in the parent process.

Output:
- `Retry summary: attempted=<N> succeeded=<S> failed=<F>`

## `simulate resume`

Resume an existing dataset.

```bash
ao-predict simulate resume <dataset> [--config <config_yaml>] [--verbose] [--threads 4] [--chunks 10]
```

Behavior:
- Validates schema.
- With `--config`, prepares `/simulation`, `/setup`, and `/options` from the YAML config exactly as `simulate init` would and compares those prepared payloads to the existing dataset.
- Fails before running if the dataset does not match the supplied config.
- Records rows that are failed before the command begins.
- Runs pending simulations.
- Retries only the rows that were already failed before the command began.
- Does not retry rows that fail during the pending pass in the same invocation.
- Uses the dataset's existing storage layout. PSF storage is controlled only by `simulate init --save-psfs`.
- With `--threads`, uses joblib/loky worker processes. The default is serial execution.
- With `--chunks`, sets the number of simulations assigned to each worker chunk.
- With `--verbose`, prints failure messages for failed simulations.

Output:
- `Resume summary: attempted=<N> succeeded=<S> failed=<F>`

## `simulate check`

Validate schema and completion status.

```bash
ao-predict simulate check <dataset> [--config <config_yaml>]
```

Behavior:
- Runs dataset schema validation.
- Reports pending and failed counts.
- With `--config`, prepares `/simulation`, `/setup`, and `/options` from the YAML config exactly as `simulate init` would and compares those prepared payloads to the existing dataset.
- Raw YAML or CSV formatting does not participate in matching; the prepared payload values are compared.

Exit code:
- `0` when dataset is valid and all simulations are successful.
- `1` when schema errors, unfinished/failed simulations, or config mismatches exist.

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
- `ee_apertures`

Most simulation-specific setup values are resolved by the simulation implementation (for TIPTOP, usually from INI).
For `TiptopSimulation`, `setup.ngs_magnitude_zeropoint` is also required.

Physical setup values use a `{value, unit}` mapping. Field names remain
unit-free:

```yaml
setup:
  ee_apertures: {value: [50.0, 100.0], unit: mas}
  ngs_magnitude_zeropoint: {value: 3.0e10, unit: photon / s}
```

### `options`

Three supported inputs:

1. Broadcast defaults (single values):
```yaml
options:
  broadcast:
    wavelength: {value: 1.65, unit: um}
    zenith_angle: {value: 20, unit: deg}
```

2. Inline table:
```yaml
options:
  table:
    columns: [wavelength, zenith_angle, atm_profile_id, r0, ngs1_r, ngs1_theta, ngs1_magnitude]
    units:
      wavelength: um
      zenith_angle: deg
      r0: m
      ngs1_r: arcsec
      ngs1_theta: deg
      ngs1_magnitude: mag
    rows:
      - [1.65, 20, 0, 0.16, 10.0, 0.0, 14.0]
      - [1.65, 25, 0, 0.14, 12.0, 30.0, 15.0]
```

3. CSV table:
```yaml
options:
  table:
    path: path/to/options.csv
    units:
      wavelength: um
      zenith_angle: deg
```

Rule:
- `options.table.path` is mutually exclusive with inline `columns` and `rows`.
- `options.table.units` must name every physical table column and must omit
  nonphysical columns such as `atm_profile_id`.
- CSV column names are lowercased by CLI parsing.
- The persisted `/options` payload always contains the NGS triplet.
- If you provide any of `ngs*_r`, `ngs*_theta`, or `ngs*_magnitude`, provide the full triplet.
- Unused star slots may be represented with `NaN` after normalization, but each slot must be either all finite or all `NaN` across radius, angle, and magnitude.
- If you omit the NGS triplet entirely, the selected simulation must supply it during options preparation.

Precedence:
- table values first
- broadcast values fill missing values
- simulation completion logic fills remaining required option keys from simulation defaults

Atmospheric input note:
- `r0` is the canonical persisted option in `/options`.
- `seeing` is accepted as an input alias (table/broadcast), converted to `r0` using `setup.atm_wavelength`, and is not persisted.
- If both `r0` and `seeing` are provided for one simulation, they must be consistent.
- In `setup.atm_profiles`, `seeing` is also accepted per profile, normalized to `r0`, and not persisted.

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
- `/stats/ee`: core `[N, M, A]`, selected by `/setup/ee_geometry`
- `/stats/fwhm`: core `[N, M]`, selected by `/setup/fwhm_summary`
- Successful runs may store `NaN` in `/stats/fwhm` when contour-based FWHM
  measurement is unrecoverable.
- Additional `/stats/*` datasets may appear when declared by the simulation in `/simulation/extra_stat_fields`; each extra stat dataset is `[N, M]`.

Every physical or scientifically dimensionless numerical HDF5 dataset carries
its canonical generic Astropy unit string in a `units` attribute. The value is
`1` for dimensionless scientific quantities.

Implemented core metric family:
- Strehl: image-domain `pixel_fit` (default) or `pixel_max`, selected by `/setup/sr_method`
- EE: fixed peak-centered image-domain aperture accumulation selected by `/setup/ee_geometry`
- FWHM: fixed native contour measurement summarized by `/setup/fwhm_summary`

Setup-level stats selectors:
- `/setup/sr_method`: dataset-level Strehl selector, `pixel_fit` or `pixel_max`
- `/setup/fwhm_summary`: dataset-level contour-summary selector, `geom`, `mean`, `max`, or `min`
- `/setup/ee_geometry`: dataset-level EE aperture selector, `ensquared` or `encircled`

Stats input note:
- Per-simulation `/options/wavelength` is required at execution time because
  the Strehl calculation builds a diffraction-limited reference PSF for each
  simulation.

Core state dataset:
- `/status/state`: `uint8[N]`

Core metadata layout:
- `/meta/pixel_scale`: per-simulation `[N]`
- `/meta/tel_diameter`: dataset-level scalar
- `/meta/tel_pupil`: dataset-level `[Ny, Nx]`

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
