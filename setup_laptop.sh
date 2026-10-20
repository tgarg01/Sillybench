#!/bin/bash
# One-time laptop setup (macOS or Linux): Python venv for the notebooks + Jupyter kernel for VS Code.
#   usage: bash setup_laptop.sh
set -euo pipefail
cd "$(dirname "$0")"
command -v gcloud >/dev/null || echo "NOTE: gcloud not found. Install it (https://cloud.google.com/sdk/docs/install), then run: gcloud auth login"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
.venv/bin/python -m ipykernel install --user --name sillybench --display-name "sillybench"
echo "Done. In VS Code open notebooks/setup.ipynb, click 'Select Kernel' (top right) ->"
echo "  'Select Another Kernel...' -> 'Jupyter Kernel...' -> 'sillybench'"
echo "  (or 'Python Environments...' -> '.venv'). Not listed? Cmd/Ctrl+Shift+P -> 'Developer: Reload Window'."
