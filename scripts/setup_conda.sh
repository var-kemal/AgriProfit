#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=agriprofit
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$THIS_DIR/.." && pwd)"

echo "Creating conda env: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda env create -f "$ROOT_DIR/environment.yml"
echo "Activating env: $ENV_NAME"
conda activate "$ENV_NAME"
echo "Env activated. To run Streamlit: streamlit run ui/app_streamlit.py"

