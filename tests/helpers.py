from __future__ import annotations

from collections.abc import Callable

from ao_predict.persistence import SimulationStore
from ao_predict.simulation import SimulationResult
from ao_predict.simulation.runner import RunSummary


def run_pending_with_callback(
    store: SimulationStore,
    run_one: Callable[[int], SimulationResult],
) -> RunSummary:
    """Run all pending simulations via callback and persist each result.

    Test helper used to exercise resume semantics independently from the
    full ``Simulation`` lifecycle.
    """
    pending = store.pending_indices()

    attempted = 0
    succeeded = 0
    failed = 0

    for index in pending:
        attempted += 1
        try:
            result = run_one(int(index))
            store.write_simulation_success(int(index), result)
            succeeded += 1
        except Exception:
            store.write_simulation_failure(int(index))
            failed += 1

    return RunSummary(attempted=attempted, succeeded=succeeded, failed=failed)
