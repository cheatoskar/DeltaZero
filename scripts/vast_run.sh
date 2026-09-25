#!/usr/bin/env bash
# Stage A on a rented GPU: download HF traces + MX blocks -> shards -> pretrain.
#
#   bash scripts/vast_run.sh [all|env|data|build|train] [HOURS] [extra pretrain args...]
#
# Resumable: downloads skip finished files, the build skips parts with a DONE marker.
# Checkpoints land in runs/ghost/{best,latest}.pt; fetch them with scripts/pull.sh.
set -euo pipefail
STAGE="${1:-all}"
HOURS="${2:-5}"
shift $(( $# > 2 ? 2 : $# )) || true
EXTRA=("$@")
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate   # Vast PyTorch images

# Size worker pools by the cgroup CPU quota, not nproc (nproc shows the host's cores).
NPROC="$(nproc)"
if [ -r /sys/fs/cgroup/cpu.max ] && read -r Q P < /sys/fs/cgroup/cpu.max && [ "$Q" != max ]; then
  NPROC=$(( Q / P )); [ "$NPROC" -lt 2 ] && NPROC=2
fi
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
run() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }
HF="https://huggingface.co/datasets/jeanmidev/trackmania-community-tracks-and-telemetry/resolve/main"
PARTS=310

if want env; then
  run environment
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
  echo "cpu quota: $NPROC"; free -g | head -2; df -h . | tail -1
  pip install -q pandas pyarrow
  # Blackwell (RTX 50xx, compute 12.0) needs a torch built for CUDA 12.8.
  if ! python -c "import torch; x = torch.ones(64, 64, device='cuda'); print('torch', torch.__version__, 'cuda ok', float((x @ x).sum()))"; then
    echo "torch cannot run on this GPU -> installing the CUDA 12.8 build"
    pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; x = torch.ones(64, 64, device='cuda'); print('torch', torch.__version__, 'cuda ok')"
  fi
fi

if want data; then
  run "download $PARTS trace parts (~19 GB) + block parts (~4.5 GB) from Hugging Face"
  mkdir -p data/hf/traces data/hf/blocks data/hf/meta
  fetch() {  # url dst: resumable, atomic
    local url="$1" dst="$2"
    [ -s "$dst" ] && return 0
    curl -sfL --retry 8 --retry-delay 5 -C - -o "$dst.part" "$url" && mv "$dst.part" "$dst" \
      || echo "FAILED $url"
  }
  export -f fetch
  fetch "$HF/tmnf/tracks_details_tmnf.parquet" data/hf/meta/tracks_details_tmnf.parquet
  for i in $(seq 1 $((PARTS - 1))); do  # HF has parts 0001..0309 (0000 and 0310 are 404s)
    p=$(printf %04d "$i")
    echo "$HF/tmnf/blocks/blocks_tmnf_part$p.parquet data/hf/blocks/blocks_tmnf_part$p.parquet"
    echo "$HF/tmnf/traces/traces_tmnf_part$p.parquet data/hf/traces/traces_tmnf_part$p.parquet"
  done | xargs -P 16 -n 2 bash -c 'fetch "$0" "$1"'
  echo "traces: $(ls data/hf/traces/*.parquet | wc -l) files, blocks: $(ls data/hf/blocks/*.parquet | wc -l) files"
  du -sh data/hf; df -h . | tail -1
fi

if want build; then
  W=$(( NPROC > 4 ? NPROC - 2 : NPROC ))
  run "build shards with $W workers (stride 2 = every 200 ms)"
  python -u tmdriver.py ghost-build --traces data/hf/traces --blocks data/hf/blocks --workers "$W" --stride 2
  du -sh data/ghost; df -h . | tail -1
  python - <<'PY'
import json, glob
s = [json.load(open(f)) for f in glob.glob('data/ghost/shards/part*/stats.json')]
tot = lambda k: sum(x.get(k) or 0 for x in s)
print(f"{len(s)} parts: {tot('maps'):,} maps, {tot('train_samples'):,} train / {tot('val_samples'):,} val samples, "
      f"runs dropped: jump {tot('jump'):,} length {tot('length'):,} time {tot('bad_time'):,}")
PY
fi

if want train; then
  DW=$(( NPROC > 12 ? 10 : NPROC - 2 )); [ "$DW" -lt 1 ] && DW=1
  ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
  run "pretrain for $HOURS h (data workers $DW, open files $(ulimit -n))"
  python -u tmdriver.py pretrain --hours "$HOURS" --workers "$DW" "${EXTRA[@]}" 2>&1 | tee -a runs/pretrain.log
fi
