# Command Reference

Run all commands from the **project root**:

```text
Transformer_MARL_EVRP/
```

Shell examples use **PowerShell** on Windows; the same commands work in bash with `python` unchanged.

---

## 1. Setup

Install dependencies (once):

```powershell
pip install -r requirements.txt
```

### GPU (recommended)

All configs default to **`device: cuda`**. Check that PyTorch sees your GPU:

```powershell
python scripts/check_gpu.py
```

If CUDA is not available, install a GPU build of PyTorch (example — adjust CUDA version to match your driver):

```powershell
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Override on the command line:

```powershell
python main_train.py --config configs/curriculum_hanif.yaml --curriculum --device cuda
```

Use `--device cpu` only for debugging small instances.

---

## 2. Unit tests

Verify the solver before training:

```powershell
python -m pytest tests/ -v
```

Single test file:

```powershell
python -m pytest tests/test_masks.py -v
```

Hanif loader tests:

```powershell
python -m pytest tests/test_hanif_loader.py -v
```

---

## 3. Dataset conversion (Hanif)

The fast loader cache (`dataset/converted_hanif_data/`, ~1 GB) is **not** in git. Convert once:

```powershell
python scripts/convert_hanif_dataset.py
```

Optional paths:

```powershell
python scripts/convert_hanif_dataset.py `
  --source dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_B_relaxed_TW `
  --output dataset/converted_hanif_data
```

Each instance folder contains: `evrp_instance.json`, `nodes.csv`, `matrices.npz`, `stations.csv`, `summary.json`.

---

## 4. Training

### Flat REINFORCE (`configs/default.yaml`)

```powershell
python main_train.py --config configs/default.yaml
```

Overrides:

```powershell
python main_train.py --config configs/default.yaml --epochs 50
python main_train.py --config configs/default.yaml --device cpu
python main_train.py --config configs/default.yaml --seed 123
```

**Fast iteration tip:** set small filters in the config:

```yaml
dataset:
  train_size_filter: ["C5"]
  eval_size_filter: ["C5"]
```

Then set both to `null` for full instances.

---

### Curriculum training

#### Phase 0 only (no training)

```powershell
python main_train.py --config configs/curriculum_hanif.yaml --phase0-only
```

#### Hanif benchmark — **recommended**

Progressive curriculum **C → RC → R** on 1000-customer instances:

```powershell
python main_train.py --config configs/curriculum_hanif.yaml --curriculum
```

`curriculum_hanif.yaml` sets `validate_during_training: false` — training runs straight through the curriculum with no gate checks or fleet-size validation pauses. Evaluate afterward with `main_evaluate.py` (see §6).

Equivalent alias:

```powershell
python main_train.py --config configs/curriculum_hanif_c_rc.yaml --curriculum
```

#### Other curriculum configs

| Command | Description |
|---------|-------------|
| `python main_train.py --config configs/curriculum.yaml --curriculum` | Classic 4-phase (C → C+R → C+R+RC → hard) |
| `python main_train.py --config configs/curriculum_progressive.yaml --curriculum` | Generic progressive C → RC → R |
| `python main_train.py --config configs/curriculum_rc.yaml --curriculum` | RC-only size curriculum |
| `python main_train.py --config configs/curriculum_c_rc.yaml --curriculum` | Legacy Schneider path |

Skip Phase 0 gate (**not recommended**):

```powershell
python main_train.py --config configs/curriculum_hanif.yaml --curriculum --force
```

#### Checkpoints saved during curriculum

- `last.pt` — final weights after training (use this when validation is disabled)
- `phase_*_best.pt` — end of each progressive class block (C, RC, R)
- With `validate_during_training: true`: also `best.pt`, `best_C.pt`, `best_R.pt`, `best_RC.pt`, `best_feasibility.pt`

Validation score (when enabled): `cost + 10000 × (1 − feasibility_rate)`

---

## 5. Live training dashboard

Run in a **second terminal** while training:

```powershell
python main_dashboard.py --config configs/curriculum_hanif.yaml
```

Open **http://localhost:8080** — auto-refresh every 3 seconds.

Port busy:

```powershell
python main_dashboard.py --config configs/curriculum_hanif.yaml --port 8081
```

Custom log path:

```powershell
python main_dashboard.py --log outputs/logs/train_log.jsonl --config configs/curriculum_hanif.yaml
```

Dashboard sections (jump links at top):

1. Total objective J(π)
2. Fleet & routing
3. Charging (time + price)
4. Waiting time
5. Service penalties / Constraint penalties
6. Solution quality / Fleet utilization
7. Per-class summaries
8. Policy optimization

---

## 6. Evaluation

Greedy decode + fleet-size search + baselines:

```powershell
python main_evaluate.py --config configs/curriculum_hanif.yaml
```

Specific checkpoint (after train-only run, use `last.pt`):

```powershell
python main_evaluate.py --config configs/curriculum_hanif.yaml --checkpoint outputs/checkpoints/last.pt
```

Force CPU:

```powershell
python main_evaluate.py --config configs/curriculum_hanif.yaml --device cpu
```

---

## 7. Output locations

| Path | Description |
|------|-------------|
| `outputs/checkpoints/` | Model checkpoints |
| `outputs/logs/train_log.jsonl` | Live training log (dashboard) |
| `outputs/logs/val_log.jsonl` | Per-class validation log |
| `outputs/results/training_metrics.csv` | Training metrics table |
| `outputs/results/validation_metrics.csv` | Validation metrics |
| `outputs/results/phase0_report.txt` | Phase 0 report |
| `outputs/results/curriculum_summary.json` | Curriculum run summary |
| `outputs/routes/` | Saved route JSON per instance |
| `outputs/figures/training/` | Training curve PNGs |
| `outputs/runs/<run_id>/` | Archived run (config + metrics) |

---

## 8. Typical full workflow (Hanif)

**Terminal 1:**

```powershell
pip install -r requirements.txt
python scripts/check_gpu.py
python scripts/convert_hanif_dataset.py
python -m pytest tests/ -v
python main_train.py --config configs/curriculum_hanif.yaml --curriculum
```

**Terminal 2:**

```powershell
python main_dashboard.py --config configs/curriculum_hanif.yaml
```

Open http://localhost:8080

**After training:**

```powershell
python main_evaluate.py --config configs/curriculum_hanif.yaml --checkpoint outputs/checkpoints/best.pt
```

---

## 9. Legacy Schneider workflow

Dataset: `dataset/Schneider/vrptw_instances/` (100-customer instances).

```powershell
python main_train.py --config configs/default.yaml
python main_dashboard.py --config configs/default.yaml
python main_evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```
