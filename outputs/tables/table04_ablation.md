# Table 4. Ablation

ALNS-FR = full recharge (30 s). ALNS-PR = partial recharge (30 s Phase I).
ALNS-PR+HCO = a **separate** 30 s ALNS-PR run, then HCO on that run's frozen sequences.
ALNS-HCO = 15 s Phase I + 15 s Phase II with frozen $K=K^*$.

## Panel A. Cross-run method means

These four rows are independent 30 s searches. **Do not read ALNS-PR → ALNS-PR+HCO as an exact fixed-route ablation.**
The wall-clock stopping criterion can yield a different Phase-I solution even at the same seed.
Fleet size $K$ differs between the stored ALNS-PR and ALNS-PR+HCO runs on **16 of 600** instance–seed pairs (original: 3/300; relaxed: 13/300). HCO itself never changes $K$ (Panel B). The larger mean $K$ of ALNS-HCO vs ALNS-PR is the shorter Phase-I budget, not an HCO fleet-size effect.

### Original TW

| Method | $K$ | distance | duration | $N_{ch}$ | energy | wait | ch. time | ch. cost | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALNS-FR | 69.97 | 47416.9 | 171088.2 | 81.02 | 23313.0 | 1245.1 | 20002.7 | 12843.4 | 30.89 |
| ALNS-PR | 69.88 | 48555.3 | 168826.1 | 79.04 | 17438.6 | 1195.0 | 14911.1 | 9652.7 | 30.90 |
| ALNS-PR+HCO | 69.87 | 48539.4 | 168752.4 | 78.40 | 17368.9 | 1184.4 | 14865.9 | 9609.5 | 30.91 |
| ALNS-HCO | 71.01 | 50082.7 | 170532.2 | 72.80 | 18157.3 | 1093.4 | 15582.5 | 10021.2 | 31.03 |

### Relaxed TW

| Method | $K$ | distance | duration | $N_{ch}$ | energy | wait | ch. time | ch. cost | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALNS-FR | 61.91 | 48244.7 | 149697.6 | 106.36 | 26697.9 | 1608.7 | 23054.8 | 14566.3 | 31.19 |
| ALNS-PR | 61.80 | 49276.6 | 148947.6 | 106.77 | 22227.9 | 1600.6 | 19195.6 | 12152.3 | 31.19 |
| ALNS-PR+HCO | 61.79 | 49267.8 | 148915.3 | 106.76 | 22234.2 | 1600.6 | 19195.3 | 12158.0 | 31.20 |
| ALNS-HCO | 62.81 | 50998.0 | 151179.5 | 103.57 | 23505.9 | 1551.7 | 20311.3 | 12846.1 | 31.32 |

## Panel B. Within-run Phase I vs final (the controlled HCO comparison)

Each row uses the `phase1_*` fields stored **inside the same run**. This is the clean test of whether HCO / Phase II changed the Phase-I route.
HCO preserves $K$ in every seed-run. Distance or charging expense changed in only a small Original-TW subset, concentrated in C1/C2, and in **none** of the Relaxed-TW seed-runs.

| TW | Method | n seed-runs | $K$ changed | $D$ or $C_{ch}$ changed | mean $K$ Phase I | mean $K$ final | mean $D$ Phase I | mean $D$ final | mean $C_{ch}$ Phase I | mean $C_{ch}$ final | families with a change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Original | ALNS-PR+HCO | 300 | 0 | 12 | 69.87 | 69.87 | 48550.3 | 48539.4 | 9644.1 | 9609.5 | C1:11, C2:1 |
| Original | ALNS-HCO | 300 | 0 | 20 | 71.01 | 71.01 | 50106.2 | 50082.7 | 10097.4 | 10021.2 | C1:16, C2:4 |
| Relaxed | ALNS-PR+HCO | 300 | 0 | 0 | 61.79 | 61.79 | 49267.8 | 49267.8 | 12158.0 | 12158.0 | — |
| Relaxed | ALNS-HCO | 300 | 0 | 0 | 62.81 | 62.81 | 50998.0 | 50998.0 | 12846.1 | 12846.1 | — |
