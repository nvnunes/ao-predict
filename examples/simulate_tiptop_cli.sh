#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXAMPLE_ID="${1:-1}"
case "${EXAMPLE_ID}" in
    1|2) ;;
    *)
        echo "Usage: $(basename "$0") [1|2]" >&2
        exit 2
        ;;
esac

CONFIG_STEM="simulate_tiptop_cli_example${EXAMPLE_ID}"
CONFIG_PATH="examples/${CONFIG_STEM}.yaml"
DATASET_PATH="examples/sims/${CONFIG_STEM}.h5"

python -m ao_predict.cli simulate init "${CONFIG_PATH}" --overwrite
python -m ao_predict.cli simulate run "${DATASET_PATH}"
python -m ao_predict.cli simulate check "${DATASET_PATH}"
