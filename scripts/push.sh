#!/usr/bin/env bash
# Upload the code (and optionally the TMX downloads) to the Vast box.
#
#   bash scripts/push.sh root@HOST PORT            code only (~1 MB)
#   bash scripts/push.sh root@HOST PORT --data     + data/bulk, data/maps, data/tmx (~1.5 GB)
#
# then on the box:
#   cd ~/TMDriverAI && tmux new -s tm && bash scripts/vast_run.sh all 5
set -euo pipefail
HOST="${1:?usage: push.sh user@host port [--data]}"
PORT="${2:?usage: push.sh user@host port [--data]}"
WITH_DATA="${3:-}"
REMOTE="${REMOTE:-~/TMDriverAI}"
cd "$(dirname "$0")/.."

FILES=(src tests tmdriver.py scripts plugin )
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
# Windows editors write CRLF; bash on the server then fails with "invalid option name".
sed -i 's/\r$//' scripts/*.sh

ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE/data"
echo "code: $(du -ch --exclude=__pycache__ "${FILES[@]}" | tail -1)"
tar czf - --exclude='__pycache__' --exclude='*.pyc' "${FILES[@]}" \
  | ssh -p "$PORT" "$HOST" "tar xzf - -C $REMOTE"

if [ "$WITH_DATA" = "--data" ]; then
  DATA=(data/bulk data/maps data/tmx data/m1/manifest.json data/calibration.json)
  echo "data: $(du -ch "${DATA[@]}" 2>/dev/null | tail -1)"
  # replays are already compressed (Gbx/LZO): plain tar, no gzip
  tar cf - "${DATA[@]}" | ssh -p "$PORT" "$HOST" "tar xf - -C $REMOTE"
fi
echo "uploaded. on the box:  cd $REMOTE && tmux new -s tm && bash scripts/vast_run.sh all 5"
