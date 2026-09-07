# ALNS-HCO method (corrected Hanif study)

## Pipeline

1. **Phase I (ALNS-PR, 15 s)** — shared \(X^{PR}_{15}\) under `outputs/shared_phase1/` with customer-route checksum. Equal 15/15 split of the common 30 s budget (not claimed optimal).

2. **Fixed-route HCO** — may only change charging stations/charges on frozen customer sequences. Non-worsening under lex \((C_{ch}, T_{wait}, T_{ch}, D)\).

3. **Phase II (15 s)** — \(K=K^*\), admit only if \(D\le(1+\varepsilon)D^{PR}\). Lex ranking \((C_{ch}, T_{wait}, T_{ch}, D)\).

## Phase II controls (same \(X^{PR}_{15}\))

| Method | Operators | HCO |
|---|---|---|
| **PhaseII-NoHCO** | Phase-II destroy + shortlist repair | No (main ablation) |
| **PhaseII-Detour** | Same + HCO-Detour | Detour only |
| **PhaseII-Heterogeneous** | Same + HCO-Heterogeneous | Heterogeneous |
| **PhaseII-Plain** | Phase-I destroy + just-enough | No (secondary: extra ordinary ALNS time) |

Main HCO ablation: PhaseII-NoHCO → PhaseII-Heterogeneous.

## HCO rules

- Explicit \((s,e_s)\) plans; never `None → full recharge`.
- Battery propagates through all prior charging stops.
- **Detour:** rank by `(detour, station_id)` only; non-worsening acceptance by
  `(D, N_ch, station-id sequence)` only — never uses \(\bar P, \bar W, r, C_{ch}, T_{wait}, T_{ch}\).
- **Heterogeneous:** top-\(k\) union of detour/price/wait/rate, dominate, then lex charge score;
  non-worsening under lex \((C_{ch}, T_{wait}, T_{ch}, D)\).
- Activity rates: `changed_calls / feasible_hco_calls`, `objective_improvements / feasible_hco_calls`.

## Frozen config

See `outputs/config/frozen_experiment_config.json`.

## Workflow

```bash
python -m pytest tests/test_hco_final_fix.py -q
python scripts/select_phase2_epsilon.py
python scripts/preserve_trusted_baselines.py   # once, before first corrected rerun
python scripts/run_final_hco_branches.py --workers 8
```
