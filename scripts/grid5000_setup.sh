#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "Grid'5000 environment ready."
echo "Dataset should be available under \$HOME/data/brain-mri-images or passed with --data-root."
