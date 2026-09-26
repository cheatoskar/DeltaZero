#!/usr/bin/env bash
# Fine-tune the driver on sim-night's exact re-simulated runs, on a training box (e.g. RTX 5090).
#
# At home:  python tmdriver.py trainpack                      -> data/trainpack.zip (~2-3 GB)
#           bash scripts/push.sh root@HOST PORT                (code)
#           scp -P PORT data/trainpack.zip root@HOST:~/TMDriverAI/data/
#           scp -P PORT runs/ghost_big2/best.pt root@HOST:~/TMDriverAI/runs/ghost_big2/
# On the box:
#           cd ~/TMDriverAI && tmux new -s ft 'bash scripts/resim_finetune.sh 3'
# Back home: bash scripts/pull.sh root@HOST PORT   (runs/ghost_resim)
#
# Steps: unpack, build the shards from the pack (no replay files needed: positions every 100 ms
# and labels from the exact per-tick actions), then continue ghost_big2 on them with the car's
# orientation switched on (exact now, not the ghost's compressed one), lower learning rate.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
HOURS="${1:-3}"
WORKERS="$(nproc)"
ulimit -n 65536 2>/dev/null || true

if [ ! -d data/trainpack/runs ]; then
  echo "=== unpacking data/trainpack.zip ($(date +%H:%M:%S)) ==="
  python - <<'PY'
import zipfile
zipfile.ZipFile('data/trainpack.zip').extractall('data')
PY
fi
echo "=== shards from the pack ($(date +%H:%M:%S)) ==="
python -u tmdriver.py ghost-replays --source trainpack --workers "$WORKERS" 2>&1 | tee -a runs/resim_shards.log
echo "=== fine-tune ghost_big2 for $HOURS h ($(date +%H:%M:%S)) ==="
python -u tmdriver.py pretrain --hours "$HOURS" --d 384 --layers 8 --heads 8 --bs 3072 --lr 1e-4 \
  --workers 10 --init runs/ghost_big2/best.pt --shards data/ghost/resim_shards:1.0 --orient \
  --seed 2 --out runs/ghost_resim 2>&1 | tee -a runs/ghost_resim.log
echo "=== done $(date +%H:%M:%S) ==="
