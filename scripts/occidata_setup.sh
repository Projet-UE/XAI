#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Occidata setup is deprecated for this repository."
echo "Delegating to the Grid'5000 setup helper instead."

exec "${SCRIPT_DIR}/grid5000_setup.sh"
