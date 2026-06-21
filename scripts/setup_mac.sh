#!/usr/bin/env bash
# Run on Mac after extracting IE540_HW_mac_export.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
# Mac: PyTorch uses MPS/CPU wheels from default index (no CUDA index-url).

echo
echo "Ready. Example:"
echo "  source .venv/bin/activate"
echo "  python evaluate.py --agent follow --episodes 5 --seed 0"
