#!/usr/bin/env bash
# Wait for train_cem_aim.py to finish, eval seeds 0-4, append CEM-Aim to leaderboard PNG.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${1:-runs/cem_aim/cpr_mpc_teacher}"
SEEDS="${2:-0:4}"
CEM_CSV="${3:-runs/cem_mpc/cem_mpc_5seed_pop256_from_episodes.csv}"
OUT_PNG="${4:-runs/comparison/agent_leaderboard_table.png}"
LOG="${RUN_DIR}/append_leaderboard.log"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== wait_and_append_cem_aim_leaderboard $(date) ==="
echo "run_dir=$RUN_DIR seeds=$SEEDS"

while pgrep -f "train_cem_aim.py.*${RUN_DIR}" >/dev/null; do
  echo "$(date): training still running..."
  sleep 120
done
echo "$(date): training finished"

MODEL="${RUN_DIR}/cem_aim_val_best.npz"
if [[ ! -f "$MODEL" ]]; then
  MODEL="${RUN_DIR}/cem_aim_train_best.npz"
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: no checkpoint in ${RUN_DIR}"
  exit 1
fi
echo "Using model: $MODEL"

.venv/bin/python evaluate_cem_aim.py \
  --model "$MODEL" \
  --seeds "$SEEDS" \
  --output-dir "$RUN_DIR"

AIM_CSV="${RUN_DIR}/per_seed_comparison.csv"
.venv/bin/python scripts/plot_agent_leaderboard_table.py \
  --seeds "$SEEDS" \
  --cem-csv "$CEM_CSV" \
  --cem-aim-csv "$AIM_CSV" \
  --out "$OUT_PNG"

echo "$(date): wrote $OUT_PNG with CEM-Aim row"
