# Multi-Agent Transformer E-VRPTW Solver

Research-grade reinforcement learning solver for the **Electric Vehicle Routing Problem with Time Windows and Recharging Stations (E-VRPTW)**.

The policy uses a **Transformer encoder–decoder** with **asynchronous multi-agent** decision-making, **hard feasibility masks**, reservation masks, partial charging, and **fleet-size search**. Training uses **centralized actor-critic** (step rewards, GAE) on Schneider **100-customer** C / R / RC instances with deterministic C→R→RC interleaving.

## Features

- Static graph Transformer encoder over depot, customers, and charging stations
- Event-driven multi-EV decoding (acting EV = smallest `current_time`)
- Hard masks for battery, capacity, time windows, anti-stranding, and reservations
- Weighted cost function **J(π)** with fleet, distance, duration, charging time, **Hanif charging price**, waiting, and feasibility penalties
- Full training on 100-customer C+R+RC with fixed interleaved order and reserved holdouts
- Live NiceGUI + Plotly training dashboard
- Per-class evaluation and macro-average reporting

## Repository layout

```
Transformer_MARL_EVRP/
├── configs/                  # YAML training config (train.yaml)
├── dataset/                  # Schneider / Hanif E-VRPTW instances
├── src/
│   ├── data/                 # Loaders, dataset splits, schedules
│   ├── env/                  # EVRPTW environment, masks, cost function
│   ├── models/               # Transformer policy + critic
│   ├── expert/               # Feasibility-first expert + demos
│   ├── training/             # Pipeline, BC, actor-critic RL
│   ├── evaluation/           # Holdout eval, metrics, fleet search
│   └── webapp/               # Live NiceGUI dashboard
├── tests/
├── main_train.py             # Full Expert → BC → RL pipeline
├── main_dashboard.py         # Training dashboard
├── COMMANDS.md               # Command reference
└── outputs/                  # Run artifacts under outputs/runs/
```

## Requirements

- Python 3.10+ (tested on 3.14 with NiceGUI compatibility shim)
- **NVIDIA GPU + CUDA PyTorch** (strongly recommended for Hanif 1000-customer runs)
- PyTorch, NumPy, Pandas, Plotly, NiceGUI, PyYAML

```powershell
pip install -r requirements.txt
```

Set `device: cuda` in `configs/train.yaml` (default). When training starts you should see a CUDA device line in the log.

Set `deterministic: true` in the config for fully reproducible runs (slightly slower on GPU).

## Quick start

Run from the **project root**. See **[COMMANDS.md](COMMANDS.md)**:

```powershell
python -m compileall src
pytest -q

python main_train.py --config configs/train.yaml --device cuda

python main_dashboard.py
```

Dashboard: **http://localhost:8080**. Edit **`configs/train.yaml`** for learning rate, cycles, model size, etc.

## Config files

| Config | Purpose |
|--------|---------|
| `configs/train.yaml` | **Primary** — Expert → BC → RL pipeline on Schneider RC `_21` |

### Training algorithm

The default command runs the **full warm-start pipeline**:

- Feasibility-first expert evaluation + demonstration generation
- Behavior cloning warm-start
- Centralized actor-critic RL fine-tuning (GAE, hard exact masks)
- Dashboard artifacts

Tune hyperparameters under `rl:` and `model:` in `configs/train.yaml`.

## Datasets

| Path | Description |
|------|-------------|
| `dataset/Hanif/…/variant_B_relaxed_TW/` | Raw Hanif `.txt` + station CSVs (in repo) |
| `dataset/converted_hanif_data/` | Pre-converted `evrp_instance.json`, `nodes.csv`, `matrices.npz`, `stations.csv` |
| `dataset/Schneider/vrptw_instances/` | Legacy 100-customer Schneider VRPTW text files |

Hanif instances use **battery 600**, per-instance **load capacity** (200 / 700 / 1000), and **log-normal charging prices** per station from the benchmark files.

## Training dashboard

The dashboard reads `outputs/logs/train_log.jsonl` and auto-refreshes every 3 seconds. Metrics are grouped into themed sections:

- Total objective **J(π)** and reward
- Fleet & routing, **Charging**, Waiting time
- Service / constraint penalties
- Solution quality, fleet utilization, per-class summaries
- Policy optimization (loss, entropy, grad norm)

Use the **Sections** jump bar to scroll to each block.

## Outputs

| Path | Content |
|------|---------|
| `outputs/runs/run_<timestamp>/` | Full pipeline run (expert, demos, BC, RL, dashboard) |
| `…/bc/bc_best_policy.pt` | Behavior-cloning warm-start checkpoint |
| `…/checkpoints/` | RL checkpoints (`best.pt`, `last.pt`, …) |
| `…/logs/train_log.jsonl` | Live RL metrics (dashboard) |
| `…/metrics/` | Training metrics CSV / cycle summaries |

## Performance notes

- Current default training uses Schneider **100-customer RC `_21`** instances.
- **GPU (`device: cuda`)** is the default.
- On **CPU** (CPU-only PyTorch), expect much slower steps.
- On **GPU**, expect substantially faster training (depends on VRAM/GPU).
- `deterministic: false` (default) enables cuDNN autotune for faster GPU training.

## License & citation

PhD research codebase. If you use the Hanif benchmark, cite the original EVRPTW charging-stations dataset authors. See `dataset/Hanif/EVRPTW_ChargingStations_Dataset/README.md`.
