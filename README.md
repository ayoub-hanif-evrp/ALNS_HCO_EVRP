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
│   ├── training/             # Actor-critic, schedules, validation
│   ├── evaluation/           # Fleet search, baselines, metrics
│   └── webapp/               # Live dashboard
├── tests/
├── main_train.py
├── main_evaluate.py
├── main_dashboard.py
├── COMMANDS.md               # Command reference (quick test / train / dashboard)
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

When training starts you should see a CUDA device line in the log.

Set `deterministic: true` in the config for fully reproducible runs (slightly slower on GPU).

## Quick start

Run from the **project root**. See **[COMMANDS.md](COMMANDS.md)** for the three commands:

```powershell
python -m compileall src
pytest -q
python main_train.py --phase0-only

python main_train.py --config configs/train.yaml --device cuda

python main_dashboard.py
```

Dashboard: **http://localhost:8080**. Edit **`configs/train.yaml`** for learning rate, cycles, model size, etc.

### Evaluate

```powershell
python main_evaluate.py --checkpoint outputs/checkpoints/last.pt --macro-average
```

See **[COMMANDS.md](COMMANDS.md)** for details on what one run does and which params to tune.

## Config files

| Config | Purpose |
|--------|---------|
| `configs/train.yaml` | **Primary** — 100-customer C/R/RC interleaved training |
| `configs/train_schneider_RC21_safe.yaml` | Explicit copy of safe RC-only baseline |
| `configs/default.yaml` | Flat actor-critic (legacy small tests) |

### Training algorithm

All training uses **centralized actor-critic**:

- Step-wise decomposed rewards aligned with EVRPTW objective
- GAE advantage estimation
- Counterfactual Q critic for multi-EV credit assignment
- Multi-sample rollout baseline (K=4 by default; mean terminal reward across K sampled rollouts)
- Potential-based reward shaping
- Edge-aware Transformer encoder (distance / travel time / energy biases)
- Local search at evaluation is **diagnostic-only** (disabled by default; distance-only 2-opt does not change reported objective)

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
- Multi-sample rollout baseline uses K=4 sampled rollouts by default (`rl.multi_sample_K`); increase only if VRAM allows.
- `deterministic: false` (default) enables cuDNN autotune for faster GPU training.

## Sanity checks

Optional environment/mask checks before training:

```powershell
python main_train.py --phase0-only
```

## License & citation

PhD research codebase. If you use the Hanif benchmark, cite the original EVRPTW charging-stations dataset authors. See `dataset/Hanif/EVRPTW_ChargingStations_Dataset/README.md`.
