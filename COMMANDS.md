# Commands

Run from the project root.

---

## Full pipeline (all at once)

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

Runs checks → demos (+ expert metrics) → BC → RL in one go.  
Expert metrics come from the same demo rollouts (no double expert pass).  
Saves everything under `outputs/runs/run_<timestamp>/`.

---

## One stage at a time

Later stages automatically continue in the latest run folder.

### 1. Sanity checks

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage checks
```

Quick internal compile/smoke checks before training.

### 2. Expert (metrics only, no demos)

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage expert
```

Rolls out the expert once and records metrics (not needed if you run demos).

### 3. Demonstrations

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage demos
```

Generates expert demos for BC and writes expert metrics from the same run.

### 4. Behavior cloning

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage bc
```

Trains the policy to imitate the expert (warm-start checkpoint).

### 5. RL fine-tuning

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage rl
```

Fine-tunes the BC policy with actor-critic RL.

---

## Dashboard

```powershell
python main_dashboard.py
```

Opens a live view of the latest run at http://localhost:8080.

---

## Optional developer checks

```powershell
python -m compileall src
```

Compiles all `src` modules to catch syntax errors.

```powershell
pytest -q
```

Runs the unit test suite.
