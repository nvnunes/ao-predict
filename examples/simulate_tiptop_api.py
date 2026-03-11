"""Minimal working example for running TIPTOP simulations via ao-predict API."""

from __future__ import annotations

from pathlib import Path

from ao_predict import (
    InitDatasetRequest,
    SetupConfig,
    SimulationConfig,
    SimulationState,
    TableOptionsConfig,
    check_dataset,
    init_dataset,
    run_simulations_by_state,
)


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    # Resolve all example inputs/outputs from this file so the script can be
    # run from either the repo root or the examples directory.
    request = InitDatasetRequest(
        dataset_path=example_dir / "sims" / "simulate_tiptop_api.h5",
        simulation=SimulationConfig(
            name="Tiptop",
            base_path=str(example_dir),
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

    # Initialize the dataset once from the simulation, setup, and options
    # configuration. This persists the input contract and creates pending runs.
    num_sims = init_dataset(request)
    dataset_path = Path(request.dataset_path)

    # Execute all simulations still marked pending, then validate the completed
    # dataset so the example fails loudly if persistence or result contracts
    # were not satisfied.
    summary = run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
    status = check_dataset(dataset_path)

    print(f"Initialized dataset: {dataset_path}")
    print(f"Simulations: {num_sims}")
    print(
        f"Run summary: attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    if not status.ok:
        print(f"Dataset check FAILED: {dataset_path}")
        for issue in status.issues:
            print(f"- {issue}")
        return

    print(f"Dataset check PASSED: {dataset_path}")
    print(f"All simulations completed successfully (N={status.num_sims}).")


if __name__ == "__main__":
    main()
