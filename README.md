# DeltaZero

An AI that drives **TrackMania Nations Forever** maps it has never seen, and then gets
faster on a map by practising it, through a TMInterface plugin.

- **Pretrained on public TMX data:** imitation learning on the driving lines of hundreds
  of thousands of ManiaExchange replays.
- **Sees the track, not pixels:** the model gets the car's recent motion, a reference line
  and the 32 nearest track blocks, so it generalises to new maps.
- **Practises in the game:** a self-imitation RL loop (cross-entropy method) runs in
  TMInterface's simulation-only mode, where the game is frozen and the physics run at a few
  hundred ticks per second.
- **Shows its plan:** it draws the planned line in the game as trigger boxes, then drives it.
  Physics and the greedy policy are both deterministic, so the visible run follows the drawn
  line exactly.
- **Re-simulation:** it replays TMX replay inputs in the game to recover the full 10 ms
  physics state, as training data for later stages. A batch mode lets the plugin play the
  inputs itself, the way TMInterface's bruteforce does.

> Status (2026-09-25): research prototype. The pretrained model picks the exact steering
> class in 78 % of held-out samples and the right steering direction in 80 %. First
> in-game runs finish maps. See [docs/STUFEN.md](docs/STUFEN.md) (German) for the
> roadmap.

The Python package is still called `tmdriver` and the plugin `TMDriver` (the project's
working name).

## How it works

```
 TMNF + TMInterface ── plugin/TMDriver (AngelScript) ── TCP 127.0.0.1:8478 ── Python (src/tmdriver)
   physics step  ─────────►  STEP (pos, rotation, velocity, inputs, …)  ─────────►  policy (DriverNet2)
                 ◄─────────  ACTION (steer, gas, brake) + commands     ◄─────────
```

| stage | data | method |
|---|---|---|
| A: pretraining | HF traces: positions of the TMX replays | behaviour cloning (transformer, 5 M params) |
| A2: replay fine-tune | top TMX replays, including car orientation | behaviour cloning |
| R: re-simulation | the same replays played in the game | full physics state every 10 ms |
| RL v1 (`improve`) | the model itself on one map | cross-entropy method / self-imitation |

Observation, every 50 ms:
- the last 10 positions (100 ms apart) in the car's heading frame, plus velocity,
  acceleration and yaw rate;
- optionally the car's forward and up vectors;
- 18 points of a reference line, 0 to 150 m ahead;
- the 32 nearest blocks within 200 m (type embedding, relative position and heading).

Outputs:
- steering over 21 bins;
- gas and brake;
- a value head (time left to the finish).

## Quick start (Windows)

Requirements: TMNF with [TMInterface](https://donadigo.com/tminterface/) 2.x (via TMLoader)
and Python 3.12.

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu126   :: GTX 10xx and newer; CPU build also works
pip install -r requirements.txt
python tmdriver.py install-plugin        :: copies plugin/TMDriver into Documents\TMInterface\Plugins
```

Models (`runs/`) and data (`data/`) are **not** in this repository. Copy them from an
existing install or train them yourself (see `docs/VAST_RUN.md`). Reading .Gbx replays
needs LZO: either the `lzo1x_*.dll` files in `src/tmdriver/gbx/lib/` (not included) or
`pip install python-lzo`.

Then:
1. Start the game and run `TMDriver_starten.bat`. It connects to the plugin; leave it open.
2. In the game's **TMDriver AI** window, open **KI-Werkzeuge**.
3. Pick a TMX id (0 means the current map), then press one of:
   - **Linie planen + fahren**: plan the line, draw it, drive it;
   - **KI trainieren**: practise the map for N rounds, then show the best run;
   - **Nachsimulieren**: re-simulate the downloaded replays.

   Each job opens its own console with live progress.

Step-by-step guide (German): [docs/HEIMRECHNER.md](docs/HEIMRECHNER.md).

## Command line

```
python tmdriver.py [--device auto|cuda|cpu] <command>
  serve            connect to the plugin; handles the in-game buttons and tool jobs
  drive            plan in simulation-only, draw the line, then drive it visibly
  improve          RL v1 on one map (--map ID --rounds N)
  resim            re-simulate TMX replays in the game (--source bulk --hours H [--per-tick])
  eval             closed-loop evaluation on held-out maps
  bulk-fetch       download awarded TMX maps + replays (rate-limited, resumable)
  ghost-build      stage A: HF traces + TMX blocks -> training shards
  ghost-replays    stage A2: TMX replays -> shards with orientation
  pretrain         train DriverNet2 (GPU)
```

## Tests

The tests run against a fake game (`tests/fake_game.py`) that mirrors the plugin's
protocol, so they need no TrackMania install:

```
python tests/test_ghost.py          # live features == training features
python tests/test_m1_fake.py        # re-simulation (batch == per-tick), dataset, training, eval
python tests/test_ghost_fake.py     # stage A pipeline end to end
python tests/test_improve_fake.py   # RL loop, drawing, plan == visible run
python tests/test_jobs_fake.py      # in-game tool buttons -> job consoles
```

## Repository layout

```
plugin/TMDriver/   TMInterface plugin (protocol 7)
src/tmdriver/      Python package: link/protocol, session, features (ghost*.py), models,
                   pretraining, RL (improve.py), re-simulation, TMX download, .Gbx reader
scripts/           GPU-box helpers (push/pull, Vast run, Wine start)
docs/              plan, data notes, guides
tests/             fake game + end-to-end tests
```

## Data and credits

- Maps and replays: [ManiaExchange (TMX)](https://tmnf.exchange). Bulk downloads are
  rate-limited and were coordinated with the TMX team. No TMX data is included here.
- Game interface: [TMInterface](https://donadigo.com/tminterface/) by donadigo.
- Inspired by [Linesight](https://github.com/Linesight-RL/linesight). No Linesight code is
  used.
