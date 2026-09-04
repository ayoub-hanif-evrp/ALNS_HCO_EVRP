# Table 2. Algorithm comparison

Per-instance means over seeds, then mean over the 60 feasible instances. Lower is better.
Displayed values are means only. CSV columns `std_k` and `std_distance` are the SD **across instances**, not across seeds.
Those SDs are large because families are heterogeneous (C1 often needs 100+ vehicles; R2 often needs about 20–30).
They are omitted from the displayed table so they are not read as algorithm instability.

## Original TW

| Method | n | Feasible | Mean $K$ | Mean distance | Mean duration | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| GI | 60 | 60/60 | 77.17 | 59123.7 | 181686.8 | 0.90 |
| VNS | 60 | 60/60 | 75.58 | 48863.0 | 179187.3 | 30.89 |
| SA | 60 | 60/60 | 75.89 | 49422.2 | 182113.9 | 30.88 |
| ALNS-FR | 60 | 60/60 | 69.97 | 47416.9 | 171088.2 | 30.89 |
| ALNS-PR | 60 | 60/60 | 69.88 | 48555.3 | 168826.1 | 30.90 |
| ALNS-HCO | 60 | 60/60 | 71.01 | 50082.7 | 170532.2 | 31.03 |

## Relaxed TW

| Method | n | Feasible | Mean $K$ | Mean distance | Mean duration | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| GI | 60 | 60/60 | 77.00 | 68841.6 | 184023.6 | 1.25 |
| VNS | 60 | 60/60 | 72.47 | 48195.7 | 166657.1 | 31.18 |
| SA | 60 | 60/60 | 72.41 | 49609.6 | 168558.7 | 31.17 |
| ALNS-FR | 60 | 60/60 | 61.91 | 48244.7 | 149697.6 | 31.19 |
| ALNS-PR | 60 | 60/60 | 61.80 | 49276.6 | 148947.6 | 31.19 |
| ALNS-HCO | 60 | 60/60 | 62.81 | 50998.0 | 151179.5 | 31.32 |
