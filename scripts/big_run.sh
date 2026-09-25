#!/usr/bin/env bash
# Bigger model on a rented 5090 (2026-09-25): env -> HF data -> shards -> train until END_UTC -> eval.
#   END_UTC=16:05 bash scripts/big_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
END_UTC="${END_UTC:-16:05}"
bash scripts/vast_run.sh env
bash scripts/vast_run.sh data
echo "=== build shards, 8 workers ($(date +%H:%M:%S)) ==="
[ -f data/ghost/shards/part0309/DONE ] || \
  python -u tmdriver.py ghost-build --traces data/hf/traces --blocks data/hf/blocks --workers 8 --stride 2
du -sh data/ghost; df -h . | tail -1
now=$(date -u +%s); end=$(date -u -d "today $END_UTC" +%s)
HOURS=$(python -c "print(round(max(0.25, ($end - $now) / 3600 - 0.2), 2))")   # 0.2 h for the final eval
echo "=== train d=384 L=8 for $HOURS h ($(date +%H:%M:%S)) ==="
ulimit -n 65536 2>/dev/null || true
python -u tmdriver.py pretrain --hours "$HOURS" --d 384 --layers 8 --heads 8 --bs 3072 --lr 7e-4 \
  --workers 10 --out runs/ghost_big 2>&1 | tee -a runs/big.log
echo "=== done $(date +%H:%M:%S) ==="
