# TMNF-C on Windows (client-free re-simulation)

DeltaZero re-simulates TMX replays without the game with
[TMNF-C](https://github.com/adonis-singh/TMNF-C) (MIT, see `LICENSE-TMNF-C`), a C
reimplementation of TrackMania Nations Forever's physics. Upstream targets Linux and Wine. This
folder holds what it takes to build and use it natively on Windows. `src/tmdriver/tmnfc.py` and
`src/tmdriver/sim_night.py` drive it.

## Contents

* `tmnfc-windows.patch` applies to TMNF-C commit `eb6be04`:
  * `src/world.c`: frees its two `aligned_alloc` blocks through `TMNF_ALIGNED_FREE`.
  * `tools/build_track`: adds a nuget.org package source. The `TMNF_LENIENT=1` mode builds maps
    that the exact builder rejects: the file's ground flag is kept, flat blocks of no zone are
    placed like classic blocks, raised water stays a plain surface, and units outside the field
    are skipped.
  * Capture tools (`oracle/`):
    * `TMNFTracer.cfg` next to the DLL can replace the environment variables, so the DLL can be
      injected into a running game.
    * `TMDriver/<stem>` map names are accepted.
    * The socket reads loop, because Windows has no `MSG_WAITALL`.
    * The memory reader runs natively.
* `compat/`:
  * `tmnf_win.h`, `sys/`, `linux/`: shims for the few Linux calls. mmap reads the file,
    futex maps to WaitOnAddress, CPU affinity is ignored, and cpuid replaces
    `__builtin_cpu_supports`. Force-included by the build.
  * `build_windows.sh`: builds the physics DLL/library and the synthetic tests with
    `zig cc` (`pip install ziglang`). No Visual Studio, no admin rights.
  * `tmnfc_batch.c`: loads one track, then runs input schedules read from stdin. It can respawn
    the car at another start first.
  * `spawn.py`: the start pose from the start block (`spawnLoc * blockLoc`, float32 in the game's
    order), bit-exact against the game on every map checked.
  * `inject.c`: loads a DLL into the running game, so the game folder stays untouched.
  * Capture and analysis: `capture_vehicle.sh`, `capture_tracks.py`, `resim_check.py`,
    `transfer_check.py`, `classify_failures.py`.

## Setup

1. Clone TMNF-C next to DeltaZero (`TMDRIVER_TMNFC` can point elsewhere), then apply the patch
   and add the compat files:
   ```
   git clone https://github.com/adonis-singh/TMNF-C && cd TMNF-C && git checkout eb6be04
   git apply ../DeltaZero/tmnfc_port/tmnfc-windows.patch
   cp -r ../DeltaZero/tmnfc_port/compat .
   ```
2. Install Python 3.12 with `ziglang`, `Pillow` and `numpy`, plus the .NET 10 SDK (for GBX.NET).
3. Extract the physics image from your own game:
   `python tools/prepare_local_assets.py --packs "C:/Program Files (x86)/TmNationsForever/Packs"`.
4. Build: `bash compat/build_windows.sh`, then build `compat/tmnfc_batch.c` the same way into
   `build/tmnfc_batch.exe`.
5. A vehicle snapshot, captured once from the game (see below), goes to
   `oracle/vehicles/LOLSPORT-Stadium.tmnfvehicle`.

Game assets, captured snapshots and built tracks are derived from your installation. Keep them
local; they are not part of this repository.

## Capturing the vehicle once

This needs a TMLoader profile running TmForever `2.12.0-compat`: same code as Nations 2.11.26,
without ASLR. The normal 2.12.0 relocates the image and breaks the memory reader. It also needs
TMInterface 2.2.1 and Linesight's `Python_Link.as` plugin, listening on port 8490. Get the plugin
from the Linesight repository yourself; it is not included here.

```
bash compat/capture_vehicle.sh <map stem in Tracks\Challenges\TMDriver> <NAME>
```

`build/win32/` needs `inject.exe`, `TMNFTracer.dll` (32-bit, `zig cc -target
x86-windows-gnu`) and a `TMNFTracer.cfg` naming the map's sha256.

**One capture drives every map.** The snapshot's track hash is re-bound to the new map, and the
car is respawned at the new start and stepped once. That gives the new map's race-time-10 state
exactly.

## Measured (2026-09-26, TMX maps with the most awards)

* Runs checked against DeltaZero's exact in-game re-simulations: 14/14 exact (0.000000 m), on
  maps other than the capture map too.
* Over the first ~150 maps, about 55-60 % of the replays are exact. A map is usually all or
  nothing. There are two causes:
  1. The offline builder orders collision entries differently from the game (fixed by a track
     dumped from the game).
  2. TMNF-C physics differs from the game, e.g. landing at 300 km/h on some ground surfaces.
     The track dumped from the game does not help there.
* Physics runs at ~120k ticks/s per core. Loading a track takes ~1-2 s, once per map.
