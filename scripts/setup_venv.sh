#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$THIS_DIR/.." && pwd)"

cd "$ROOT_DIR"
python3 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Venv activated. To run Streamlit: streamlit run ui/app_streamlit.py"

