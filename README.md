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

> Status (2026-09-26): research prototype. The best pretrained model (`ghost_big2`, 14.7 M
> parameters) picks the exact steering class in 78.6 % of held-out samples and the right
> steering direction in 81 %. In-game runs finish short race maps; RL with PPO over several
> game instances is being tested. Imitation has plateaued, so the next gains come from RL,
> physics inputs from re-simulated replays, and route planning from the map itself.

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

The best checkpoints (`runs/*/best.pt`) are in this repository; data (`data/`) is not.
Maps and replays are fetched from TMX when needed. Reading .Gbx files uses LZO: the
`lzo1x_*.dll` files in `src/tmdriver/gbx/lib/` (not included), `pip install python-lzo`,
or a slower pure-Python fallback.

Then:
1. Start the game and run `Start_DeltaZero.bat`. It connects to the plugin; leave it open.
2. In the game's **DeltaZero** window, pick a TMX id (0 means the current map) and whether
   the AI may use the reference line, then press one of:
   - **Drive**: plan the line, draw it, drive it;
   - **Train**: practise the map for N rounds, showing new best runs;
   - **Re-simulate replays**: play TMX replays in the game and store the physics state.

   Each job opens its own console with live progress.

Without the game window, double-click a launcher instead (then `Start_DeltaZero.bat` must
not run at the same time): `Drive.bat`, `Train.bat`, `Train_RL.bat`, `Resimulate.bat`,
`Resimulate_TestMaps.bat`, `Check_Hold.bat`. Every job can be stopped with Ctrl+C; Train and
RL then save and show the best run.

Setup on Windows with an Nvidia GPU: Python 3.12, `pip install torch --index-url
https://download.pytorch.org/whl/cu126` (cu126 still supports Pascal cards such as the
GTX 1080 Ti), `pip install numpy pandas pyarrow`, then `python tmdriver.py install-plugin`.
Keep TMInterface up to date in TMLoader (an old version queued maps without loading them).
Further game instances ("helpers") take the next free plugin ports and share the work of
re-simulation, Train and RL.

## Command line

```
python tmdriver.py [--device auto|cuda|cpu] [--hold N] <command>
  serve            connect to the plugin; handles the in-game buttons and tool jobs
  drive            plan in simulation-only, draw the line, then drive it visibly
  improve          RL v1 on one map (--map ID --rounds N [--line])
  rl               RL v2 (PPO) on one map (--map ID --iterations N [--line])
  resim            re-simulate TMX replays in the game (--maps IDS | --pool | --source bulk)
  tmx-pool         map pool from the TMX search (most awarded maps in a time range)
  launch           start helper game instances
  check-hold       check that --hold gives identical runs in the real game
  eval             closed-loop evaluation on held-out maps
  bulk-fetch       download awarded TMX maps + replays (rate-limited, resumable)
  ghost-build      stage A: HF traces + TMX blocks -> training shards
  ghost-replays    stage A2: TMX replays -> shards with orientation
  pretrain         train DriverNet2 (GPU)
```

Without a reference line (the default) the model sees only the track blocks and its own
motion; the fastest TMX replay, if there is one, only judges progress during training.

## Tests

The tests run against a fake game (`tests/fake_game.py`) that mirrors the plugin's
protocol, so they need no TrackMania install:

```
python tests/test_ghost.py          # live features == training features
python tests/test_m1_fake.py        # re-simulation (batch == per-tick), dataset, training, eval
python tests/test_ghost_fake.py     # stage A pipeline end to end
python tests/test_improve_fake.py   # RL v1, two instances, Ctrl+C, drawing, plan == visible run
python tests/test_rl.py             # RL v2 maths (GAE, action distribution, PPO direction)
python tests/test_rl_fake.py        # RL v2 on two instances, action hold == per-tick
python tests/test_course.py         # progress without a line, stall rules, respawn skip
python tests/test_jobs_fake.py      # in-game tool buttons -> job consoles
```

## Repository layout

```
plugin/TMDriver/   TMInterface plugin (protocol 12)
src/tmdriver/      Python package: link/protocol, session, fleet (several instances), features
                   (ghost*.py), models, pretraining, RL v1 (improve.py) and v2 (rl.py),
                   re-simulation, TMX download, .Gbx reader
scripts/           GPU-box helpers (push/pull, Vast run, Wine start), course_check.py
tests/             fake game + end-to-end tests
```

## Data and credits

- Maps and replays: [ManiaExchange (TMX)](https://tmnf.exchange). Bulk downloads are
  rate-limited and were coordinated with the TMX team. No TMX data is included here.
- Game interface: [TMInterface](https://donadigo.com/tminterface/) by donadigo.
- Inspired by [Linesight](https://github.com/Linesight-RL/linesight). No Linesight code is
  used.
