# Multi-Agent Transformer E-VRPTW Solver

Research-grade reinforcement learning solver for the **Electric Vehicle Routing Problem with Time Windows and Recharging Stations (E-VRPTW)**.

The policy uses a **Transformer encoder–decoder** with **asynchronous multi-agent** decision-making, **hard feasibility masks**, reservation masks, partial charging, and **fleet-size search**. Training uses **centralized actor-critic** (step rewards, GAE) on Schneider **RC _21** instances with flat cyclic scheduling.

## Features

- Static graph Transformer encoder over depot, customers, and charging stations
- Event-driven multi-EV decoding (acting EV = smallest `current_time`)
- Hard masks for battery, capacity, time windows, anti-stranding, and reservations
- Weighted cost function **J(π)** with fleet, distance, duration, charging time, **Hanif charging price**, waiting, and feasibility penalties
- Flat cyclic training on 13 Schneider RC `_21` train instances (3 held out for eval)
- Phase 0 sanity checks (optional via `--phase0-only`)
- Live NiceGUI + Plotly training dashboard with grouped metric sections
- Per-class validation and multi-checkpoint saving (`best_C.pt`, `best_R.pt`, …)

## Repository layout

```
Transformer_MARL_EVRP/
├── configs/                  # YAML training / evaluation configs
├── dataset/
│   ├── Hanif/                # Hanif benchmark (variant B relaxed TW) — source .txt
│   ├── Schneider/            # Legacy Solomon/Schneider VRPTW instances
│   └── converted_hanif_data/ # Generated fast loader (not in git — run convert script)
├── scripts/
│   ├── convert_hanif_dataset.py
│   └── check_gpu.py
├── src/
│   ├── data/                 # Loaders, Hanif parser/converter, dataset splits
│   ├── env/                  # EVRPTW environment, masks, cost function
│   ├── models/               # Transformer policy
│   ├── training/             # Actor-critic, flat schedule, validation, Phase 0
│   ├── evaluation/           # Fleet search, baselines, metrics
│   └── webapp/               # Live dashboard
├── tests/
├── main_train.py
├── main_evaluate.py
├── main_dashboard.py
├── COMMANDS.md               # Full command reference
└── outputs/                  # Logs, checkpoints, figures (generated)
```

## Requirements

- Python 3.10+ (tested on 3.14 with NiceGUI compatibility shim)
- **NVIDIA GPU + CUDA PyTorch** (strongly recommended for Hanif 1000-customer runs)
- PyTorch, NumPy, Pandas, Plotly, NiceGUI, PyYAML

```powershell
pip install -r requirements.txt
```

### GPU setup

Training defaults to **`device: cuda`** in all configs. Verify PyTorch sees your GPU **before** a long run:

```powershell
python scripts/check_gpu.py
```

If it fails, you likely have the **CPU-only** PyTorch wheel. Install a CUDA build matching your driver (example for CUDA 12.4):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Pick the correct wheel at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

When training starts you should see:

```text
Curriculum training on device: cuda — NVIDIA GeForce RTX … (… GB)
```

With GPU, Hanif steps are typically **several× faster** than CPU (~45–90 s/step on CPU).

Set `deterministic: true` in the config for fully reproducible runs (slightly slower on GPU).

## Quick start (Hanif benchmark — recommended)

Run from the **project root**.

### 1. Convert dataset (one time, ~1 GB local cache)

The converted NPZ folders are **not** stored in git. Regenerate them:

```powershell
python scripts/convert_hanif_dataset.py
```

This writes `dataset/converted_hanif_data/` (60 instances, 1000 customers each, per-station charging rates and prices).

### 2. Run tests

```powershell
python -m pytest tests/ -v
```

### 3. Train (Schneider RC _21)

```powershell
python main_train.py
```

Trains on **13 RC _21** instances in cyclic order (configurable `cycles` in `configs/train.yaml`). Use `--overfit` for a single-instance sanity check.

**Terminal 2** — live dashboard:

```powershell
python main_dashboard.py
```

Open **http://localhost:8080**. Edit **`configs/train.yaml`** for learning rate, training cycles, model size, etc.

### 4. Evaluate

```powershell
python main_evaluate.py --checkpoint outputs/checkpoints/last.pt
```

See **[COMMANDS.md](COMMANDS.md)** for details on what one run does and which params to tune.

## Config files

| Config | Purpose |
|--------|---------|
| `configs/train.yaml` | **Primary** — Schneider RC _21 flat actor-critic |
| `configs/train_schneider_RC21_safe.yaml` | Explicit copy of safe baseline config |
| `configs/default.yaml` | Flat actor-critic (legacy small tests) |

### Training algorithm

All training uses **centralized actor-critic**:

- Step-wise decomposed rewards aligned with EVRPTW objective
- GAE advantage estimation
- Counterfactual Q critic for multi-EV credit assignment
- POMO-style multi-rollout baseline (K=4 by default)
- Potential-based reward shaping
- Edge-aware Transformer encoder (distance / travel time / energy biases)
- Local search repair at evaluation (2-opt distance polish)

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
| `outputs/checkpoints/` | `best.pt`, `best_C.pt`, `best_R.pt`, `best_RC.pt`, `last.pt`, … |
| `outputs/logs/train_log.jsonl` | Live training log (dashboard) |
| `outputs/logs/val_log.jsonl` | Per-class validation |
| `outputs/results/training_metrics.csv` | Flat metrics table |
| `outputs/results/phase0_report.txt` | Environment & mask checks |
| `outputs/runs/<run_id>/` | Archived run config + metrics |

## Performance notes

- Hanif instances have **~1000 customers + ~333 stations** (~1334 nodes).
- **GPU (`device: cuda`)** is the default — run `python scripts/check_gpu.py` first.
- On **CPU** (CPU-only PyTorch), expect **~45–90 s per training iteration**.
- On **GPU**, expect **much faster** steps (often 5–20× vs CPU, depending on VRAM/GPU).
- POMO uses K=4 rollouts per step by default (`rl.pomo_K`); increase only if VRAM allows.
- `deterministic: false` (default) enables cuDNN autotune for faster GPU training.

## Phase 0 checks

Before training, optional Phase 0 checks validate loaders, matrices, masks, battery/capacity/TW logic, and depot return:

```powershell
python main_train.py --phase0-only
```

## License & citation

PhD research codebase. If you use the Hanif benchmark, cite the original EVRPTW charging-stations dataset authors. See `dataset/Hanif/EVRPTW_ChargingStations_Dataset/README.md`.
