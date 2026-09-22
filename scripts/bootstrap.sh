#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is required but not found in PATH."
  exit 1
fi

env_prefix="$repo_root/.conda"

install_source() {
  case "$1" in
    git+*|http://*|https://*)
      conda run --prefix "$env_prefix" python -m pip install "$1"
      ;;
    *)
      conda run --prefix "$env_prefix" python -m pip install -e "$1"
      ;;
  esac
}

if [ ! -d "$env_prefix" ]; then
  echo "[bootstrap] Creating conda environment at $env_prefix..."
  conda create -y --prefix "$env_prefix" python=3.12 pip
fi

echo "[bootstrap] Installing package with dev+docs extras..."
if [ -n "${AO_PREDICT_HYBRID_SOURCE:-}" ]; then
  install_source "$AO_PREDICT_HYBRID_SOURCE"
fi
install_source "${AO_PREDICT_NGS_PHOTOMETRY_SOURCE:-git+https://github.com/nvnunes/ngs-photometry.git}"
conda run --prefix "$env_prefix" python -m pip install -e ".[dev,docs]"

echo "[bootstrap] Configuring git hooks path..."
git config core.hooksPath .githooks

echo "[bootstrap] Running quick checks..."
conda run --prefix "$env_prefix" python -m pytest -q
conda run --prefix "$env_prefix" mkdocs build --strict

echo "[bootstrap] Done."
echo "Environment: $env_prefix"
echo "Git hooks path: $(git config --get core.hooksPath)"
