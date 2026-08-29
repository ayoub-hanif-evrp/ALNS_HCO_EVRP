# Comparison pack for the paper

**Cite this file, not published objective values from other EVRP papers.**
Those papers use different datasets, charging physics, waiting models, and objectives. There is **no published numerical table** on this 1000-customer Hanif Homberger set.

Two comparisons exist:

1. **Literature positioning** (model/method identity). No K or distance copied from Schneider, Keskin, Nafstad, Wang, etc. onto the Hanif 1000-customer set.
2. **Internal numerical table** on Hanif: Constructor, ALNS-style, VNS-style, Classical, Proposed. Same 120 instances.
3. **External literature BKS** on the 56 Schneider 100-customer instances (`outputs/schneider/`). CARE-FI is run in FR and PR modes and compared to published \(K,D\) (Erdelić & Carić 2022 consolidation of Schneider, Goeke, Hiermann, Keskin, Schiffer). Lexicographic: min \(K\) then min \(D\). Distance gap only when \(\Delta K=0\). This does **not** use the internal ALNS-style/VNS-style codes.

Call the competitors **ALNS-style** and **VNS-style**. Do not write “Schneider et al.’s VNS–TS” or “Keskin & Çatay’s ALNS” as if the codes were reproduced.

**Do not write:** “CARE-FI outperformed Keskin and Çatay’s ALNS by 15.8% / 28.8%.” That result does not exist. The gap is CARE-FI versus lightweight matched neighbourhoods.

Wang et al. (2026, *Transportation Research Part C*, two-phase lexicographic ALNS) already minimise fleet then routing. Do not claim “two-phase fleet-first EVRPTW” as CARE-FI’s novelty. The distinction is 1000-customer CARE intensification plus route-preserving heterogeneous charging refinement (\(\mathbb{E}[W]\), \(\mathbb{E}[P]\), rate) on Hanif.

**Proposed in `outputs/proposed/` is the 28 Aug Stage-2 regeneration** (balanced pool, \(\Phi\) ranking, route decomposition) on frozen Classical. Scientific PASS, 120/120 feasible.

Primary ranking is lexicographic **fleet size \(K\)**, then distance. A competitor with more vehicles and lower station wait is **not** better.

Source CSVs: `outputs/tables/table1_overall.csv`, `table2_family.csv`, `table4_method_comparison.csv`, `literature_competitors.csv`.

---

## 1. What other papers did (positioning only)

| Study | Method | Partial charge | Heterogeneous stations | Station waiting | Price | Exact piece | Scale in that paper |
|---|---|---|---|---|---|---|---|
| Schneider et al. (2014) | Hybrid VNS–TS | Basic recharge | Limited | No | No | No | Canonical EVRPTW, ≤100 customers |
| Keskin & Çatay (2016) | ALNS | Continuous partial | No | No | No | MILP formulation | ≤100 customers |
| Keskin et al. (2021) | Simulation + ALNS | Yes | Queue uncertainty | Stochastic | No | Simulation | 100-customer EVRPTW |
| Froger et al. (2022) | ILS + B&C | Variable | Multiple technologies | Capacity-induced | No | B&C assembler | Benchmark EVRP |
| Jia et al. (2024) | Hybrid memetic | Yes | — | — | — | No | 1000-customer, but **pickup-delivery** |
| Nafstad et al. (2025) | Branch-price-and-cut | Yes | Heterogeneous nonlinear | — | — | Exact BP&C | ≤100 cust., 21 CS |
| Wang et al. (2026) | MILP + AILS | Yes | Heterogeneous | Time-dependent | TOU | Exact fixed-route scheduler | 100-cust. controlled; 1100–1500 real |
| Wang et al. (2026, TRC) | Two-phase lexicographic ALNS | Yes | — | — | — | — | EVRPTW (not Hanif 1000) |
| **CARE-FI (this paper)** | Fleet-first + route-preserving charging refinement | Discrete portions | Rate + E[wait] + E[price] | Heterogeneous expected wait | Heterogeneous expected price | Route-level validation planned | **120 × 1000 customers** |

Safe sentence:

> Prior EVRPTW heuristics were developed on Schneider-scale instances (typically up to 100 customers). No published algorithm reports fleet size on this 1000-customer Hanif station set. We therefore re-run ALNS-style and VNS-style neighbourhoods under the same protocol, and position CARE-FI against the literature on modelling features rather than copied objective values.

Unsafe: quoting Schneider 2014 or Keskin 2016 K/distance as if they were obtained on Hanif.

---

## 2. Numerical comparison — overall means (60+60, all feasible)

Means of instance-level metrics. **\(K\) is the headline criterion.**

### Original TW (tight windows)

| Method | Feasible | Mean \(K\) | Mean distance | Mean duration | Mean station wait | Mean charge time | Mean charge expense | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constructor | 60/60 | 94.82 | 60,487 | 220,283 | 408 | 12,400 | 6,324 | 55 |
| VNS-style | 60/60 | 94.73 | 59,512 | 221,352 | 458 | 12,919 | 6,708 | 187 |
| ALNS-style | 60/60 | 93.58 | 59,829 | 219,457 | 480 | 13,442 | 7,044 | 136 |
| **Classical (Stage 1)** | **60/60** | **78.82** | **55,119** | **193,319** | 716 | 19,188 | 10,300 | 122 |
| **Proposed (Stage 1+2)** | **60/60** | **78.82** | **54,782** | **190,177** | 689 | 12,204 | 5,935 | 159 |

Proposed runtime = Classical + refinement.

### Relaxed TW

| Method | Feasible | Mean \(K\) | Mean distance | Mean duration | Mean station wait | Mean charge time | Mean charge expense | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constructor | 60/60 | 94.82 | 60,487 | 220,283 | 408 | 12,400 | 6,324 | 20 |
| VNS-style | 60/60 | 94.68 | 57,502 | 223,489 | 483 | 12,730 | 6,659 | 186 |
| ALNS-style | 60/60 | 91.53 | 58,729 | 217,484 | 541 | 14,264 | 7,586 | 119 |
| **Classical (Stage 1)** | **60/60** | **65.20** | **51,088** | **160,768** | 924 | 20,945 | 11,946 | 96 |
| **Proposed (Stage 1+2)** | **60/60** | **65.20** | **50,596** | **157,645** | 848 | 14,648 | 7,205 | 145 |

**Constructor Relaxed TW is not an independent search.** Intensification was off, so the A→B warm start was kept. Relaxed constructor numbers duplicate Original TW. Do not call constructor B a relaxed-TW optimum. ALNS-style still only reaches mean \(K=91.53\) on Relaxed TW.

### Fleet reduction of CARE-FI Stage 1 vs matched baselines

| Versus | Original TW | Relaxed TW |
|---|---|---|
| Constructor | 94.82 → 78.82 (**−16.9%**, −16.0 EVs) | 94.82 → 65.20 (**−31.2%**, −29.6 EVs) — constructor B is a copy of A |
| VNS-style | 94.73 → 78.82 (**−16.8%**, −15.9 EVs) | 94.68 → 65.20 (**−31.1%**, −29.5 EVs) |
| ALNS-style | 93.58 → 78.82 (**−15.8%**, −14.8 EVs) | 91.53 → 65.20 (**−28.8%**, −26.3 EVs) |

### Per-instance fleet wins (lower \(K\) is better)

| Comparison | Original TW | Relaxed TW |
|---|---|---|
| Classical / Proposed vs Constructor | fewer EVs **60/60** | fewer **60/60** |
| Classical / Proposed vs VNS-style | fewer **60/60** | fewer **60/60** |
| Classical / Proposed vs ALNS-style | fewer **60/60** | fewer **58/60**, tie **2/60**, never worse |
| Proposed vs Classical | same \(K\) **60/60** | same \(K\) **60/60** |

CARE-FI never uses more vehicles than a matched baseline on any of the 120 instances.

---

## 3. How to talk about wait / charging cost vs competitors

Constructor, ALNS-style, and VNS-style often show **lower station wait and lower charging expense** than Classical. That is expected: they keep **~15–30 extra EVs**, so routes are shorter and charge less.

After Stage 2, Proposed’s charging expense (5,935 Original / 7,205 Relaxed) is in the same band as those high-fleet methods, **while keeping Classical’s smaller fleet**.

Do not write: “ALNS has better waiting than CARE-FI.” Write: ALNS-style uses 15–26 more EVs; waiting is not the primary criterion.

---

## 4. Proposed vs Classical (same fleet — the Stage-2 claim)

Same \(K\) on every instance. Aggregate relative change \(100(\bar y/\bar x-1)\):

| KPI | Original TW | Relaxed TW | Wins / 60 (Orig., Rel.) |
|---|---:|---:|---|
| \(K\) | 0.00% | 0.00% | 0 / 0 worse |
| Distance | **−0.61%** | **−0.96%** | **60/60**, **60/60** |
| Duration | **−1.63%** | **−1.94%** | **60/60**, **60/60** |
| Station waiting | **−3.77%** | **−8.23%** | 46/60, 54/60 |
| Charging time | **−36.40%** | **−30.06%** | **60/60**, **60/60** |
| Charging expense | **−42.38%** | **−39.69%** | **60/60**, **60/60** |
| Charged energy | **−37.85%** | **−33.04%** | **60/60**, **60/60** |
| Customer early waiting | +5.15% | +7.31% | worse (not in \(\Phi\)) |

Mechanism (do not claim faster chargers):

| Indicator | Original | Relaxed |
|---|---:|---:|
| Cost / charged energy | **−7.28%** | **−9.93%** |
| Station wait / visit | **−7.03%** | **−8.53%** |
| Charge time / energy | +2.34% | +4.44% |

---

## 5. Family fleet sizes (mean \(K\))

### Original TW

| Family | Constructor | ALNS-style | VNS-style | Classical / Proposed |
|---|---:|---:|---:|---:|
| C1 | 122.5 | 122.0 | 122.5 | **115.2** |
| C2 | 67.8 | 66.8 | 67.7 | **48.3** |
| R1 | 134.8 | 132.4 | 134.7 | **117.1** |
| R2 | 56.0 | 55.3 | 56.0 | **42.2** |
| RC1 | 118.8 | 117.2 | 118.5 | **103.7** |
| RC2 | 69.0 | 67.8 | 69.0 | **46.4** |

### Relaxed TW

| Family | Constructor | ALNS-style | VNS-style | Classical / Proposed |
|---|---:|---:|---:|---:|
| C1 | 122.5 | 120.1 | 122.5 | **102.1** |
| C2 | 67.8 | 65.2 | 67.7 | **42.2** |
| R1 | 134.8 | 129.2 | 134.4 | **96.6** |
| R2 | 56.0 | 53.9 | 56.0 | **29.3** |
| RC1 | 118.8 | 114.8 | 118.5 | **92.4** |
| RC2 | 69.0 | 66.0 | 69.0 | **28.6** |

The largest fleet gaps vs ALNS-style are on **C2, R2, RC2** (long-horizon families), especially under Relaxed TW (e.g. RC2: 66.0 → 28.6).

Stage-2 charging-time gains (Proposed vs Classical, Original TW) are largest on C1 (−65.1%), R1 (−54.8%), RC1 (−52.1%), C2 (−50.0%); smaller on RC2 (−24.0%) and R2 (−17.8%). See `table2_family.csv`. On Original TW, mean station wait rose on R2 (+3.2%) while still falling overall.

---

## 6. Protocol caveats the paper must keep

- **Single seed (0)** for all methods. CARE-FI Stage 1 is deterministic; ALNS-style and VNS-style are stochastic. Ten-seed competitor repeats are **not** in this pack. Do not claim statistical dominance over stochastic metaheuristics beyond this one matched run.
- **180 s** wall-clock per instance for Constructor / ALNS-style / VNS-style / Classical. Proposed adds up to 90 s of Stage 2. VNS-style often hits the cap (~187 s).
- ALNS-style is destroy/repair, **not** Keskin & Çatay 2016 (no adaptive operator weights, no continuous recharge quantity).
- VNS-style is relocate/swap/2-opt, **not** Schneider 2014 VNS+tabu (tabu list is unused).
- ILS, LNS×GTS, HMA, Froger B&C, Nafstad BP&C, Wang AILS / two-phase ALNS were **not** run. Do not invent their Hanif numbers.
- Proposed is the 28 Aug Stage-2 regeneration. Do not cite the superseded 26 Aug −40.02% / −37.69% expense figures.

---

## 7. Sentences you can paste

**Positioning.** CARE-FI is a fleet-first heuristic for 1000-customer EVRPTW with heterogeneous charging rate, expected station waiting, and expected price. It is not an exact method and is not a joint routing-and-charging metaheuristic in the sense of Wang et al. (2026).

**Fleet.** Under a matched 180 s budget, CARE-FI Stage 1 uses 78.82 EVs on Original TW and 65.20 on Relaxed TW, versus 93.58 / 91.53 for ALNS-style and 94.73 / 94.68 for VNS-style, with 60/60 feasibility. The fleet is smaller than every matched baseline on every Original-TW instance and on 58/60 Relaxed-TW instances (two ALNS-style ties).

**Charging.** Holding that fleet fixed, Stage 2 reduces charging expense by 42.4% (Original) and 39.7% (Relaxed), mainly by charging less energy, with additional reductions in cost per energy (−7.3% / −9.9%) and wait per visit. Charging time per energy does not fall.

**What we did not show.** We do not claim to beat Schneider or Keskin on their 100-customer tables, and we do not claim that ALNS-style / VNS-style are those published algorithms. We do not report published K from other papers on this dataset. We do not claim two-phase fleet-first as a novel framework (Wang et al. 2026 already do this).
