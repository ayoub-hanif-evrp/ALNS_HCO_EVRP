# Commands

Run everything from the **project root**.

## 1. Setup (once)

```powershell
pip install -r requirements.txt
python scripts/check_gpu.py
```

## 2. Train — Schneider RC `_21` safe baseline (default)

```powershell
python main_train.py
```

**Phase A only:** actor-critic + step rewards + GAE (no Q critic, POMO, shaping, edge encoder).

- **13** RC `_21` train instances × **50** cycles = **650** updates
- Held out: `rc101_21`, `rc104_21`, `rc208_21` (never loaded during training)
- Schneider has **16** RC `_21` instances total (not 23 like R class)

Longer run (~1000 updates):

```powershell
python main_train.py --cycles 77
```

Overfit sanity test (one instance):

```powershell
python main_train.py --overfit
```

Explicit safe config:

```powershell
python main_train.py --config configs/train_schneider_RC21_safe.yaml
```

## 3. Live training monitor (second terminal)

```powershell
python main_dashboard.py
```

Open **http://localhost:8080**

## 4. Evaluate later (held-out test instances)

```powershell
python main_evaluate.py --config configs/train.yaml --checkpoint outputs/checkpoints/last.pt
```

---

**Config:** `configs/train.yaml` or `configs/train_schneider_RC21_safe.yaml`
