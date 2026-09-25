#!/usr/bin/env bash
# Phase 2 of the big run: when phase 1 (tmux "big") is done, continue from its best weights
# with a fresh cosine schedule (lower peak lr, other data order) until END_UTC.
#   END_UTC=19:55 bash scripts/big_phase2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
END_UTC="${END_UTC:-19:55}"
until grep -q "^=== done" runs/big_run.out 2>/dev/null; do sleep 30; done
now=$(date -u +%s); end=$(date -u -d "today $END_UTC" +%s)
HOURS=$(python -c "print(round(max(0.25, ($end - $now) / 3600 - 0.2), 2))")
echo "=== phase 2: continue from runs/ghost_big/best.pt for $HOURS h ($(date +%H:%M:%S)) ==="
ulimit -n 65536 2>/dev/null || true
python -u tmdriver.py pretrain --hours "$HOURS" --d 384 --layers 8 --heads 8 --bs 3072 --lr 3e-4 \
  --workers 10 --init runs/ghost_big/best.pt --seed 1 --out runs/ghost_big2 2>&1 | tee -a runs/big2.log
echo "=== phase 2 done $(date +%H:%M:%S) ==="
