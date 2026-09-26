#!/usr/bin/env bash
# Windows build of TMNF-C with zig cc (pip install ziglang), no Linux, no admin rights.
# Needs local/game-mask.bin (tools/prepare_local_assets.py --packs ".../TmNationsForever/Packs").
# Local source changes: src/world.c frees its two aligned_alloc blocks with TMNF_ALIGNED_FREE.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$LOCALAPPDATA/Programs/Python/Python312/python.exe}"
mkdir -p build/generated build/obj
"$PY" tools/prepare_game_mask.py header local/game-mask.bin build/generated/tmnf_local_game_mask.h
FLAGS="-target x86_64-windows-gnu -std=gnu11 -O2 -Icompat -include compat/tmnf_win.h -Isrc -Ipython
 -Ibuild/generated -DTMNF_LOCAL_GAME_MASK=1 -ffp-contract=off -fno-fast-math -fno-math-errno
 -fno-slp-vectorize -mevex512 -mxsave"
"$PY" -m ziglang cc $FLAGS -shared -o build/tmnf_physics.dll src/*.c python/tmnf_env_ffi.c -lapi-ms-win-core-synch-l1-2-0
for f in src/*.c; do "$PY" -m ziglang cc $FLAGS -c "$f" -o "build/obj/$(basename "$f" .c).o"; done
"$PY" -m ziglang ar rcs build/libtmnf_physics.a build/obj/*.o
for t in physics_smoke x87_fp_identity vehicle_aux_smoke vehicle_curve_smoke gate_observation; do
  "$PY" -m ziglang cc $FLAGS -UNDEBUG "tests/$t.c" build/libtmnf_physics.a -lapi-ms-win-core-synch-l1-2-0 -o "build/$t.exe"
  "./build/$t.exe" > "build/$t.log" 2>&1 && echo "ok   $t" || echo "FAIL $t"
done
