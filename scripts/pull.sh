#!/usr/bin/env bash
# Fetch the trained checkpoints (runs/ghost, runs/ghost_ft), logs and re-simulated replays.
#
#   bash scripts/pull.sh root@HOST PORT
set -euo pipefail
HOST="${1:?usage: pull.sh user@host port}"
PORT="${2:?usage: pull.sh user@host port}"
REMOTE="${REMOTE:-~/TMDriverAI}"
cd "$(dirname "$0")/.."
for run in ghost ghost_ft; do
  mkdir -p runs/$run
  for f in best.pt latest.pt history.json; do
    scp -P "$PORT" "$HOST:$REMOTE/runs/$run/$f" runs/$run/ 2>/dev/null || echo "(no $run/$f yet)"
  done
done
# re-simulated replays (small): merged into data/resim
mkdir -p data/resim
scp -r -P "$PORT" "$HOST:$REMOTE/data/resim/." data/resim/ 2>/dev/null || echo "(no resim data)"
scp -P "$PORT" "$HOST:$REMOTE/runs/pretrain.log" runs/ 2>/dev/null || true
ls -la runs/ghost runs/ghost_ft
