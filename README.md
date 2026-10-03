# DeltaZero

An autonomous driving AI for **TrackMania Nations Forever** that learns to race on unseen tracks from track geometry and physics, and self-improves through reinforcement learning.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![TMInterface](https://img.shields.io/badge/TMInterface-2.x-brightgreen.svg)](https://donadigo.com/tminterface/)
[![Physics](https://img.shields.io/badge/Simulator-TMNF--C-purple.svg)](https://github.com/adonis-singh/TMNF-C)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Highlights

- **Track Perception, Not Raw Pixels:** The policy observes the local 3D world: the 32 nearest track blocks (types, relative offsets, and orientations), a 10-step car motion history in the ego-heading frame (velocities, accelerations, yaw rate), and forward route waypoints. It operates without computer vision overhead (~14 ms CPU latency) and generalizes immediately to unseen maps.
- **Client-Free Headless Physics (TMNF-C):** Includes a native Windows port of [TMNF-C](https://github.com/adonis-singh/TMNF-C) for exact, bit-deterministic physics re-simulation at thousands of ticks per second without launching the game client (`sim-night`, `virtual`).
- **Learned Topological Route Planner (v3):** Solves track progression from start through checkpoints to the finish across a 2.5D rasterized collision mesh using Dijkstra with learned transit costs (`route_cost_model.json`). Enables zero-shot racing on complex community tracks without human replays.
- **Batched Vectorized RL (`rl-vec`):** High-throughput PPO reinforcement learning simulating dozens of cars in parallel inside headless TMNF-C worlds with automatic policy checkpointing (`model_best.pt`) and learning rate decay on divergence.
- **Pretrained Checkpoints Shipped In-Repo:** Comes out-of-the-box with `runs/ghost_resim/best.pt`—a 14.7M parameter transformer model fine-tuned on exact TMNF-C physics re-simulations.
- **Deterministic 3D Plan Visualization:** Plans its driving line in simulation, renders the planned trajectory directly in the game as 3D trigger boxes, and then drives it visibly in real time.

---

## System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          TrackMania Nations Forever           │
                  │              + TMInterface 2.x                │
                  └───────────────────────┬───────────────────────┘
                                          │ AngelScript Plugin (TCP 127.0.0.1:8478)
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DeltaZero (Python)                                     │
│                                                                                        │
│   ┌───────────────────────────┐                ┌───────────────────────────────────┐   │
│   │     Perception State      │                │          Policy Model             │   │
│   │ • 10-step ego motion      │───────────────►│  DriverNet2 (Transformer Policy)  │   │
│   │ • 32 nearest 3D blocks    │                │ • 21-bin Steering Distribution    │   │
│   │ • Route Planner Waypoints │                │ • Throttle & Braking Heads        │   │
│   └───────────────────────────┘                │ • Value Head (Time-to-Finish)     │   │
│                 ▲                              └─────────────────┬─────────────────┘   │
│                 │                                                │                     │
│                 │                                                ▼                     │
│   ┌─────────────┴─────────────┐                ┌───────────────────────────────────┐   │
│   │  Learned Route Planner v3 │                │         Action Execution          │   │
│   │ • 2.5D Dijkstra Search    │                │ • Single / Multi-tick Hold        │   │
│   │ • Logistic Cost Model     │                │ • Deterministic In-Game Replay    │   │
│   └───────────────────────────┘                └───────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ Headless Virtual Game Bridge
                  ┌───────────────────────┴───────────────────────┐
                  │       TMNF-C Native C-Engine Simulator        │
                  │   Fast re-simulation & batched parallel RL    │
                  └───────────────────────────────────────────────┘
```

---

## Driving Models & Progression

| Stage | Data Source | Method | Status |
|---|---|---|---|
| **A: Pretraining** | Millions of TMX replay positions | Behaviour cloning (Transformer, 5M params) | Pretrained (`runs/ghost/best.pt`) |
| **A2: Replay Fine-tune** | High-rated TMX replays + car orientation | Behaviour cloning (14.7M params) | Pretrained (`runs/ghost_big2/best.pt`) |
| **R: Re-simulation** | Exact 10 ms physics re-simulations via TMNF-C | Fine-tuning on full physics state | **Default Driver** (`runs/ghost_resim/best.pt`) |
| **RL v1 (`improve`)** | In-game simulation rollouts on one map | Cross-entropy method / self-imitation | Production ready |
| **RL v2 (`rl` / `rl-vec`)** | Multi-instance game fleet or batched TMNF-C | PPO with Generalized Advantage Estimation (GAE) | Production ready |

---

## Quick Start (Windows)

### 1. Requirements

- Python 3.12 (64-bit)
- TrackMania Nations Forever with [TMInterface 2.x](https://donadigo.com/tminterface/) (installed via [TMLoader](https://donadigo.com/tmloader/))
- Nvidia GPU (recommended, CUDA 12.6 supported back to Pascal / GTX 10xx series; CPU execution is fully supported)

### 2. Setup

Clone the repository and install dependencies:

```bat
git clone https://github.com/cheatoskar/DeltaZero.git
cd DeltaZero

:: Install PyTorch (CUDA 12.6 or CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu126

:: Install Python requirements
pip install -r requirements.txt

:: Install the AngelScript plugin into your TMInterface Plugins folder
python tmdriver.py install-plugin
```

> **Note on Model Checkpoints:** The repository already contains the best checkpoints (`runs/*/best.pt`). Raw training datasets (`data/`) and cache files are fetched or built on demand.

### 3. Running with the In-Game Interface

1. Start **TrackMania Nations Forever** via TMLoader (ensure the `TMDriver` plugin is active in TMInterface).
2. Double-click `Start_DeltaZero.bat` (or run `python tmdriver.py serve`).
3. In the game window, press the `~` / F3 key to view the **DeltaZero** plugin panel:
   - **Drive:** Plans the optimal racing line in simulation-only mode, visualizes it on track, and executes the run.
   - **Train:** Practises the loaded map using reinforcement learning, showing new record attempts live.
   - **Re-simulate replays:** Replays known leaderboard inputs to extract ground-truth 10 ms physics states.

### 4. Standalone Launchers (Without In-Game Menu)

When running automated jobs without interacting with the plugin UI, use the provided batch scripts:

- `Drive.bat` — Computes a plan and drives the active track.
- `Train.bat` — RL v1 training loop on the current map.
- `Train_RL.bat` — RL v2 (PPO) optimization across running instances.
- `Resimulate.bat` — Re-simulates TMX replays for the current map.
- `Sim_Night.bat` — Headless overnight re-simulation loop on TMNF-C.
- `Check_Hold.bat` — Validates that action-holding preserves bit-exact physics determinism.

---

## CLI Reference

Run `python tmdriver.py <command>` for full control:

| Command | Description |
|---|---|
| `serve` | Connects to the TMInterface plugin and serves in-game UI buttons. |
| `drive` | Plans a line in simulation-only mode, draws 3D triggers, and drives visibly. |
| `rl` | Runs RL v2 (PPO) on one map across all active TMInterface game instances. |
| `rl-vec` | Runs batched vectorized PPO without launching the game (using TMNF-C). |
| `improve` | Practises a map with self-imitation / cross-entropy RL (RL v1). |
| `sim-night` | Headless overnight crawler: downloads TMX maps and re-simulates in TMNF-C. |
| `route-learn` | Evaluates and trains the route planner cost model (`route_cost_model.json`). |
| `virtual` | Starts virtual game instances emulating TMInterface over TMNF-C physics. |
| `tmx-pool` | Queries the TMX search API for top-awarded maps to build a test pool. |
| `check-hold` | Tests whether `--hold 5` matches 1-tick precision in the game engine. |
| `fetch-tmx <ID>` | Fetches a map and its top replays directly from ManiaExchange. |

---

## Headless Simulation with TMNF-C

DeltaZero includes native support for client-free simulation via `tmnfc_port/`:

1. Uses the C physics engine from [TMNF-C](https://github.com/adonis-singh/TMNF-C) patched for Windows (`tmnfc_port/tmnfc-windows.patch`).
2. Enables running headless training and re-simulation loops on servers or local background threads without DirectX/Wine rendering overhead.
3. Automatically computes exact spawn locations, checkpoints, and finish triggers.

See [tmnfc_port/README.md](tmnfc_port/README.md) for build instructions and patch details.

---

## Running the Unit Tests

All unit tests run against an emulated game protocol (`tests/fake_game.py`) and do **not** require TrackMania to be installed:

```bash
# Test perception features and neural policy
python tests/test_ghost.py

# Test PPO math (GAE, action distributions, KL divergence)
python tests/test_rl.py

# Test course progress tracking, stall rules, and respawns
python tests/test_course.py

# Test end-to-end multi-instance RL and action holding
python tests/test_rl_fake.py
```

---

## Credits & Acknowledgements

- **[ManiaExchange (TMX)](https://tmnf.exchange)** for providing the public replay and map archive.
- **[TMInterface](https://donadigo.com/tminterface/)** by donadigo for the TrackMania automation and simulation interface.
- **[TMNF-C](https://github.com/adonis-singh/TMNF-C)** by adonis-singh for the C physics reimplementation.
- **[Linesight](https://github.com/Linesight-RL/linesight)** for inspirational research in TrackMania reinforcement learning.
