#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "Grid'5000 environment ready."
echo "Classification data should be available under \$HOME/data/brain-mri-images or passed with --data-root."
echo "autoPET FDG data can be prepared from a NIfTI source root with scripts/autopet_prepare_fdg.py."
