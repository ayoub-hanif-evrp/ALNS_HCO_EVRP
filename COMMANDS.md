# Commands

Run from the **project root**.

Study: **Schneider type-2** · anti-drift RL (strong BC-reg every step, early-stop patience 2, low entropy).

---

## Train (full Expert → BC → RL → eval)

```powershell
python main_train.py --config configs/train.yaml --device cuda
```

---

## Resume RL only (after BC is done)

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage rl
```

---

## Test (holdout)

```powershell
python main_train.py --config configs/train.yaml --device cuda --stage eval
```

---

## Dashboard

```powershell
python main_dashboard.py
```

Open **http://127.0.0.1:8080**
