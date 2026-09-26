#!/usr/bin/env bash
# Capture a vehicle snapshot on Windows from the running game (TMLoader profile TMNFC =
# 2.12.0-compat, no ASLR; Python_Link on port 8490). build/win32/TMNFTracer.cfg must name the
# map's sha256. Usage: compat/capture_vehicle.sh <map stem in Challenges\TMDriver> <OUT_NAME>
set -euo pipefail
cd "$(dirname "$0")/.."
MAP="$1"; OUT="$2"
PY="${PY:-$LOCALAPPDATA/Programs/Python/Python312/python.exe}"
export PYTHONIOENCODING=utf-8
DZ="$HOME/Downloads/DeltaZero"

restart_game() {
  powershell -c "Stop-Process -Name TmForever -Force -ErrorAction SilentlyContinue; Start-Sleep 3; Start-Process -FilePath \"\$env:LOCALAPPDATA\\TMLoader\\TMLoader.exe\" -ArgumentList 'run','TmForever','TMNFC','\"/configstring=set custom_port 8490\"'"
  until powershell -c "if (Get-NetTCPConnection -State Listen -LocalPort 8490 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done
}
game_pid() { powershell -c "(Get-Process TmForever).Id" | tr -d '\r'; }
focus_later() { ( sleep 4; cd "$DZ"; "$PY" -c "import sys; sys.path.insert(0,'src'); from tmdriver.session import focus_game_window; focus_game_window($1)" ) & }

echo "== phase capture (tracer) on $MAP"
restart_game
rm -rf cap/vehicle-traces; mkdir -p cap/vehicle-traces
./build/win32/inject.exe "$(cygpath -w "$PWD/build/win32/TMNFTracer.dll")"
focus_later "$(game_pid)"
(cd oracle && "$PY" determinism_test.py --port 8490 --map "TMDriver/$MAP" --ticks 10 --output-dir ../cap/vehicle-probe | tail -2)
wait
"$PY" oracle/tracer/extract_model6_vehicle.py cap/vehicle-traces/007C3E80_CSceneVehicleCar_ComputeForcesModel6.bin "cap/$OUT-phase.tmnfvehicle"

echo "== vehicle graph (memory reader) on $MAP"
restart_game
focus_later "$(game_pid)"
(cd oracle && "$PY" dump_vehicle_snapshot.py --port 8490 --map "TMDriver/$MAP" --phase-input "../cap/$OUT-phase.tmnfvehicle" --output "vehicles/$OUT.tmnfvehicle" --report "../cap/$OUT.md" | tail -1)
wait
ls -la "oracle/vehicles/$OUT.tmnfvehicle"
