# Schneider source and route validation

## Full matrix + parameter fidelity (this round)
Checked all 56 instances for **g, v, Q, C, r**, charging rate (=1/g), and **all** distance/time/energy matrix entries against Euclidean source.
**OK: 56 / 56.**
Convention: unrounded Euclidean float32; travel=d/v; energy=d*r.
Detail: `FINAL_TABLES/Schneider_source_full_matrix_fidelity.csv`.

## Hard-customer insertion diagnosis
For each of the 12 hard instances, the repeatedly missing customer admits a **station-assisted solo route** under full recharge, but fails depot–customer–depot on battery (round-trip energy exceeds Q). Instance-level unserved=1 across methods/seeds is therefore classified as a **search/construction limitation**, not a converter bug and not a proof of official-instance infeasibility.

| Instance | Customer | TW width | Solo D–C–D | Assisted patterns feasible |
|---|---|---:|---|---:|
| r101_21 | C65 | 10 | no (battery_arrival node=0) | 2 |
| r102_21 | C65 | 10 | no (battery_arrival node=0) | 2 |
| r103_21 | C65 | 10 | no (battery_arrival node=0) | 2 |
| r105_21 | C65 | 30 | no (battery_arrival node=0) | 3 |
| r106_21 | C65 | 30 | no (battery_arrival node=0) | 3 |
| r108_21 | C64 | 30 | no (battery_arrival node=0) | 4 |
| rc101_21 | C75 | 30 | no (battery_arrival node=0) | 3 |
| rc103_21 | C75 | 158 | no (battery_arrival node=0) | 4 |
| rc104_21 | C75 | 158 | no (battery_arrival node=0) | 4 |
| rc105_21 | C75 | 29 | no (battery_arrival node=0) | 3 |
| rc106_21 | C75 | 60 | no (battery_arrival node=0) | 4 |
| rc107_21 | C75 | 74 | no (battery_arrival node=0) | 4 |

Unserved-customer IDs: `Schneider_hard_unserved_customer_ids.csv`.
120 s diagnostic (prior): 72/72 still unserved=1.
