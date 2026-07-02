# Commands

Run everything from the **project root**.

## 1. Setup (once)

```powershell
pip install -r requirements.txt
python scripts/check_gpu.py
```

## 2. Train — Schneider R `_21` (default)

```powershell
python main_train.py
```

Flat training (no curriculum): **20** R full instances × **20 passes** = **400 epochs**.  
Held out for your later testing: `r101_21`, `r104_21`, `r209_21` (not used during training).

**Outputs:**

| File | Content |
|------|---------|
| `outputs/logs/train_log.jsonl` | Per-epoch training metrics |
| `outputs/results/training_metrics.csv` | Same metrics in CSV |
| `outputs/checkpoints/last.pt` | Latest model |
| `outputs/runs/run_*/holdout_instances.json` | Train vs held-out instance lists |

Evaluate later when you want:

```powershell
python main_evaluate.py --config configs/train.yaml --checkpoint outputs/checkpoints/last.pt
```

## 3. Live training monitor (recommended — second terminal)

While training runs, open the NiceGUI dashboard in another terminal:

```powershell
python main_dashboard.py
```

Open **http://localhost:8080** — auto-refreshes every 3 seconds.

Shows the same training graph groups as the static figures (learning, feasibility, operations, coordination, cycles). Uses **training metrics only** and excludes held-out instances.

Moving-average window: `training.dashboard_ma_window` in config (default **20**).

## 4. Schneider curriculum (optional)

```powershell
python main_train.py --config configs/curriculum.yaml --curriculum
```

Cyclic C → R → RC training on Schneider instances.

---

**Config:** `configs/train.yaml` — Schneider path, explicit 20 train names, `passes: 20`, no `curriculum:` block.
