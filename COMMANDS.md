# Commands

Run from the project root. Install once: `pip install -r requirements.txt`

## Full training (~102 steps, ~1.5–2 h on GPU)

```powershell
python main_train.py
```

Edit **`configs/train.yaml`** before running (learning rate, cycles, model size, etc.).

## Live dashboard (second terminal)

```powershell
python main_dashboard.py
```

Open **http://localhost:8080**

---

### What one training run does

This is **not** one pass over the full dataset. It runs a **progressive curriculum** on the **80% train split** only:

| Block | Instances | Cycles | Steps |
|-------|-----------|--------|-------|
| C small (10 customers) | 6 | ×3 | 18 |
| C full (1000 customers) | 7 | ×2 | 14 |
| RC small | 7 | ×2 | 14 |
| RC full | 8 | ×2 | 16 |
| R small | 10 | ×2 | 20 |
| R full | 10 | ×2 | 20 |
| **Total** | | | **102** |

Each step = one instance, one REINFORCE update. Order: **C → RC → R**.

Outputs: `outputs/checkpoints/last.pt`, `outputs/logs/train_log.jsonl`, `outputs/figures/training/`

### Main knobs in `configs/train.yaml`

| Parameter | Section | Effect |
|-----------|---------|--------|
| `learning_rate` | `training` | Adam step size |
| `entropy_coef` | `training` | Exploration |
| `cycles_by_class` | `curriculum` | How many times each instance is repeated (≈ total training length) |
| `embedding_dim`, `encoder_layers` | `model` | Network size |
| `device` | top | `cuda` or `cpu` |
| `train_size_filter` | `dataset` | Restrict to named instances for quick tests |

### After training (optional)

```powershell
python main_evaluate.py --checkpoint outputs/checkpoints/last.pt
```

### One-time setup (if dataset missing)

```powershell
python scripts/convert_hanif_dataset.py
python scripts/check_gpu.py
```

---

## Fixed mini curriculum (15 instances, 3000 updates)

Exactly **5 C + 5 RC + 5 R** instances, fixed for the whole run. Same 15 instances every cycle.

```powershell
python main_train.py --config configs/train_fixed15.yaml --curriculum
python main_dashboard.py --config configs/train_fixed15.yaml
python main_evaluate.py --config configs/train_fixed15.yaml --checkpoint outputs/checkpoints/last.pt
```

Resume (same manifest, same 15 instances):

```powershell
python main_train.py --config configs/train_fixed15.yaml --curriculum --resume outputs/checkpoints/last.pt --run-dir outputs/runs/<run_id>
```

Edit **`configs/train_fixed15.yaml`**: `cycles_by_class` (200 per class = 3000 updates), `dataset.fixed_train_instances`, `learning_rate`.

Manifest: `outputs/runs/<run_id>/fixed_train_manifest.json`
