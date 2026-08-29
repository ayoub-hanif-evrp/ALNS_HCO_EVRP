# Architecture and methodology

This document is the source of truth for the **paper method**. It describes the solver that produced the corrected 120-instance results, not the Transformer / BC / DAgger / RL code that still exists in the repository and is **not** used.

**Headline experiment archive (cite these numbers):**  
`outputs/` — Classical solutions in `outputs/classical/`, Proposed in `outputs/proposed/`.

**Cite figures from:** `outputs/figures/`  
**Cite tables from:** `outputs/tables/`

Do not cite old wait_μ-as-minutes fleet means (80.07 / 65.15). Those archives were removed.

**Do not freeze figures or the Results section yet.** Current `outputs/` numbers are the first full 120 under corrected wait. Stage 2 is being corrected (balanced station pool, Φ-normalized visit ranking, explicit route decomposition). Regenerate Proposed before writing the final paper.

---

## 1. What the paper is about

The problem is a **1000-customer electric vehicle routing problem with time windows and heterogeneous charging stations** (EVRPTW).

There are two coupled decisions:

1. **Fleet and routing.** How many EVs, which customers on which routes, in which order.
2. **Charging.** Where to charge, how much to charge, and therefore how much time, expected station waiting, and expected charging cost is incurred.

The method is **two-stage and fleet-first**:

- **Stage 1 (Classical)** finds a feasible plan that minimises the number of EVs, then distance.
- **Stage 2 (Proposed)** freezes that fleet size \(K^*\) and refines only charging decisions to reduce an operational charging objective \(\Phi\).

The scientific claim is:

> A fleet-first solver establishes a feasible low-fleet routing plan. A heterogeneous-station refinement then exploits expected station waiting, expected charging price, and charging rate **without increasing the number of EVs**.

It is **not** a claim that “two-phase fleet-first EVRPTW” is new as a framework. Wang, Adulyasak, and Cordeau (2026) already use lexicographic fleet sizing then routing with ALNS and partial recharge. CARE-FI’s distinction is **1000-customer CARE intensification** plus **route-preserving heterogeneous charging refinement** (station-specific \(\mathbb{E}[W]\), \(\mathbb{E}[P]\), rate) on the Hanif set.

It is **not** a claim that adding waiting and price to a single fleet-search objective finds fewer vehicles. That single-search design was tried and failed (tiny mixed deltas; fleet sometimes worse).

---

## 2. What this paper is not

The writing agent must not describe any of the following as the paper method:

| Excluded | Why |
|---|---|
| Transformer, attention routing, MARL | Disabled. Config `pipeline` and `hybrid_search.neural_guidance.enabled` are false. |
| Behaviour cloning, DAgger, RL fine-tuning | Not run. |
| Chance-constrained / sampled waiting | Feasibility uses deterministic \(\mathbb{E}[W_s]\). \(\sigma\) is not sampled per visit. |
| CVaR / risk-aware \(\Phi\) | Implemented, **not** the paper method. Final run has \(\rho=0\). A CVaR pilot hurt Relaxed-TW reliability. |
| Density, carbon, battery degradation, driver wage, parking, station capacity, renewables | Explicitly excluded from \(\Phi\). |
| Independent Proposed fleet search | Abandoned. Stage 2 inherits Classical routes and freezes \(K^*\). |
| Old 80.07 / 65.15 fleet means | Those used incorrect waiting (`wait_mu` treated as minutes). |

---

## 3. Benchmark

**Dataset.** Hanif Homberger-scale EVRPTW, 60 instance names, two time-window settings:

- **Variant A / Original TW** — tight Solomon/Homberger windows.  
  Folder: `dataset/Hanif/converted_hanif_original_tw/`
- **Variant B / Relaxed TW** — the same geography and demand with relaxed windows.  
  Folder: `dataset/Hanif/converted_hanif_relaxed_tw/`

Families: **C1, C2, R1, R2, RC1, RC2**, ten instances each (`*_10_1` … `*_10_10`).

Each instance has:

- 1000 customers
- 1 depot
- 333 charging stations
- heterogeneous charging **rate**, log-normal **price**, log-normal **waiting**
- battery capacity \(Q = 600\)
- energy consumption rate \(h = 1\) (so energy on an arc equals distance)
- partial charging allowed, portions \(\{0.25, 0.50, 0.75, 1.00\}\)
- depot charging allowed
- stations reusable

**Protocol name:** EXTENDED (`src/evaluation/hanif_protocols.py`).

Seed **0**. Wall-clock budget **180 s** per instance for Stage 1. Stage 2 budget **90 s** per instance. Eight workers. Variant B may warm-start from the Variant A incumbent of the same name (A→B), but B is still solved and reported separately.

Always report Original TW and Relaxed TW **separately**. Never pool them into one mean.

---

## 4. Station waiting and price semantics

CSV fields `wait_mu`, `wait_sigma`, `price_mu`, `price_sigma` are **log-normal parameters**, not minutes and not raw prices.

For \(X \sim \mathrm{LogNormal}(\mu,\sigma)\):

\[
\mathbb{E}[X] = \exp\!\left(\mu + \tfrac12\sigma^2\right).
\]

Implemented in `src/env/lognormal.py` as `expected_lognormal`.

- **Feasibility and timing** add \(\mathbb{E}[W_s]\) when a vehicle arrives at station \(s\). Never CVaR, never a random draw.
- **Charging expense** in \(\Phi\) is \(\sum e_{s}\,\mathbb{E}[P_s]\) over charging visits with energy \(e_s>0\).
- **Station waiting in \(\Phi\)** is \(\sum \mathbb{E}[W_s]\) over those visits (one expected wait per visit, not per unit energy).

Customer early waiting (arriving before a time window) is recorded but **is not in \(\Phi\)**.

CVaR exists only as an unused sensitivity mode (`risk_aware_heterogeneous`, blend \(\rho\)). Do not call the model chance-constrained.

---

## 5. Two-stage architecture

```
                    Hanif instance (A or B)
                              |
                              v
              +--------------------------------+
              |  STAGE 1 — Classical           |
              |  min lex(hard, unserved, K,    |
              |          distance, duration)   |
              +--------------------------------+
                              |
                    feasible routes, K*
                    freeze D_class, T_class, C_class
                              |
                              v
              +--------------------------------+
              |  STAGE 2 — Station refinement  |
              |  K frozen (never add an EV)    |
              |  min Φ, accept only if Φ falls |
              |  and hard feasibility holds    |
              +--------------------------------+
                              |
                    Proposed solution
                    exact replay metrics
```

Stage 1 does **not** see price or waiting in its ranking tuple.  
Stage 2 does **not** reopen fleet minimisation.

Proposed runtime reported in tables is

\[
t_{\text{Proposed}} = t_{\text{Classical}} + t_{\text{refine}}.
\]

That is intentional and honest.

---

## 6. Stage 1 — Classical fleet optimisation

**Code.** `scripts/run_hanif_fps.py` → `src/search/deterministic_fps.py` → `src/search/fleet_search.py` → `src/search/hanif_fleet_intensify.py`.

**Mode.** `paper_objective.mode = classical_distance`.

### 6.1 Ranking (all minimised)

\[
(\text{hard violations},\; \text{unserved customers},\; K,\; \text{distance},\; \text{duration})
\]

Hard violations: battery, capacity, time windows, depot return. A solution is feasible only if hard \(= 0\) and every customer is served exactly once.

There is **no** \(\Phi\) in Stage 1. Charging price is not in the Classical ranking (`charging_price_in_objective: false`).

### 6.2 Construction

On 1000-customer instances the constructor is **dense nearest-route insertion** (`src/search/fast_construct.py`, `construction_backend: dense`), not a full exact best-insertion over every position. It produces a feasible (or near-feasible) large fleet quickly.

Fleet bounds: a capacity lower bound plus an adaptive upper start around \(n/6.5\) vehicles, not a padded maximum fleet.

### 6.3 Fleet descent

The search reduces \(K\) while remaining feasible:

1. Probe an upper feasible fleet.
2. Descend toward the lower bound with warm starts from a larger fleet.
3. Compact unused vehicles.
4. Route elimination: remove a route and reinsert its customers.
5. Repair by regret / greedy insertion (`src/search/repair.py`). Geographic `_nearest_stations` is used **here** (distance-only). That is acceptable in Stage 1 because Stage 1 is not claiming station intelligence.

On this scale the solver **skips** full route-level charging DP, set partitioning, and generic ALNS unless a competitor flag is set.

### 6.4 Hanif fleet intensification

After a feasible dense plan, `intensify_hanif_fleet` spends up to about 90 s (or 55% of remaining budget) repeatedly trying to delete vehicles.

A route is a candidate for deletion if it looks easy to absorb. The constraint-aware risk score (CARE) used to rank elimination candidates is, after min–max normalisation of each term:

\[
\mathrm{risk} = 0.40\,\mathrm{size} + 0.20\,\mathrm{overlap} + 0.20\,(1-\mathrm{slack}) + 0.10\,\mathrm{load} + 0.10\,\mathrm{charge\ burden}.
\]

Low risk = small route, overlapping neighbours, loose time windows, light load, few charges. Customers are reinserted with bounded regret repair. A deletion is kept only if the Classical lex tuple improves.

This is **fleet compression**, not station optimisation.

### 6.5 Exact accept

Every incumbent is verified by exact replay (`src/search/route_replay.py`, `src/search/exact_checks.py`): sequences, charge labels, energy on every arc, waiting split, SoC, legal portions in \((0,1]\).

---

## 7. Stage 2 — Heterogeneous charging refinement

**Code.** `scripts/run_final_expected.py` → `src/search/station_refine.py` → `src/search/station_candidates.py`.

**Input.** The finished Classical routes for that instance, and frozen normalisers

\[
D_{\mathrm{class}},\quad T_{\mathrm{station,class}},\quad C_{\mathrm{charge,class}}
\]

measured on those same routes (`src/evaluation/objective_reference.py`). The old “loose constructor” normaliser is forbidden.

**Invariant.** \(K_{\text{Proposed}} = K_{\text{Classical}} = K^*\). The search never adds an EV. Customer-to-route assignment is not redesigned; only charging visits and charge amounts on existing routes change.

### 7.1 Objective \(\Phi\)

Mode `expected_heterogeneous`, risk blend \(\rho = 0\).

Let \(D\) be total distance. Over charging visits with energy \(e_s>0\):

\[
T_{\mathrm{chg}} = \sum_s \frac{e_s}{r_s},\qquad
W_{\mathrm{station}} = \sum_s \mathbb{E}[W_s],\qquad
T_{\mathrm{station}} = T_{\mathrm{chg}} + W_{\mathrm{station}},
\]

\[
C_{\mathrm{charge}} = \sum_s e_s\,\mathbb{E}[P_s].
\]

\(r_s\) is the station charging rate. Then

\[
\Phi = \frac{D}{D_{\mathrm{class}}} + \frac{T_{\mathrm{station}}}{T_{\mathrm{station,class}}} + \frac{C_{\mathrm{charge}}}{C_{\mathrm{charge,class}}}.
\]

The three terms are unweighted. A Classical solution has \(\Phi \approx 3\). Proposed should go below 3 if charging burden falls without inflating distance too much.

**Ranking in Stage 2:**

\[
(\text{hard},\; \text{unserved},\; K,\; \Phi,\; \text{distance},\; \text{duration}).
\]

Metadata: `charging_price_in_objective: true`, `expected_wait_in_objective: true`.

### 7.2 Candidate generation (this is the methodological point)

Stage 1 repair ranks stations by **detour / distance only**. Stage 2 does not.

For a charge positioned between nodes \(i\) and \(j\), `operational_station_pool` builds five ranked lists:

- lowest expected wait
- lowest expected price
- highest charging rate
- smallest detour
- best combined min–max score on a detour-capped subset (detour \(\le \max(3\times\text{nearest}, \text{nearest}+40)\))

Those lists are merged by **round-robin / quota** (`balanced_union`). Sequential concat-then-truncate is forbidden, because wait/price/rate can otherwise spend the entire shortlist and drop detour and combined stations.

A cheap or fast station can enter the pool even if it is not among the \(k\) nearest. Exact replay later rejects geographically impossible detours.

Visits to try first are ranked by their **normalized contribution to \(\Phi\)** (charging time plus expected wait over \(T^0\), plus energy times expected price over \(C^0\)). Raw minutes are never added to raw prices.

### 7.3 Neighbourhoods and route decomposition

Under exogenous station attributes, reusable stations, and frozen customer sequences, \(\Phi\) is additive over routes. Stage 2 therefore refines **one route at a time** (still confirmed by full-solution exact replay). Neighbourhoods on a route, in order:

1. **Station substitution** plus charge portion in \(\{0.5, 0.75, 1.0\}\) (highest \(\Phi\)-burden visits first).
2. **Portion-only** changes on the current station.
3. **Insert** an operationally attractive station between consecutive customer/depot anchors.
4. **Delete** a redundant station visit.

Customers are not moved between routes. Routes are scanned round-robin so a long first route cannot consume the whole budget.

### 7.4 Accept rule

A trial is kept only if exact replay shows:

- hard-feasible
- 100% served
- used vehicles \(\le K^*\)
- \(\Phi\) **strictly** smaller than the incumbent

Route-level replay is used as a filter; improving trials are confirmed on the full solution. Energy, wait, price, and time are always those of the exact replay, not of the estimator.

---

## 8. Exact metrics reported

Replay reports, and paper tables should use:

- used vehicles \(K\)
- distance
- energy consumed (sum of `energy_matrix` on every arc; with \(h=1\) this equals distance)
- charged energy
- charging visits
- charging time
- station waiting \(\sum \mathbb{E}[W_s]\)
- customer early waiting (not in \(\Phi\))
- SoC / minimum SoC
- expected charging expense \(\sum e_s\mathbb{E}[P_s]\)
- runtime (Classical; Proposed = Classical + refine)

---

## 9. Experimental protocol

| Item | Value |
|---|---|
| Instances | All 60 names × 2 TW = 120 |
| Development / gate set | 18 names: families C1–RC2 × indices 1,2,3 (both TW) |
| Seed | 0 |
| Stage 1 budget | 180 s |
| Stage 2 budget | 90 s |
| Workers | 8 |
| Battery \(Q\) | 600 |
| Objective Stage 1 | Classical lex \((K, D)\) |
| Objective Stage 2 | \(\Phi\) with Classical normalisers |
| Risk-aware / CVaR | Not run |
| Full 120 started only after | 18-instance scientific gate PASS |

**Gate (must match the scientific decision):** 100% feasible and served; \(K\) never worse on any instance; on **each** TW at least one of station wait / charging time / expense improves by \(\ge 2\%\); none of those three worsens by \(\ge 1\%\).

The previous bug (PASS on a 0.59% charging-time blip while wait and price got worse) is fixed in `src/evaluation/validation_gate.py`.

**Validation 18 result:** PASS, then full 120 was run.  
**Full 120 result:** scientific PASS after regenerating Stage 2 (28 Aug) on frozen Classical. `outputs/VALIDATION.json`. Source: `outputs/refinement_v2/` copied into `outputs/proposed/`.

Classical Stage 1 is unchanged (26 Aug). Proposed is the 28 Aug Stage-2 implementation (balanced candidate pool, \(\Phi\)-burden ranking, route decomposition).

---

## 10. Headline numerical results (full 120, Stage 2 regenerated 28 Aug)

All 120/120 feasible, 100% served, **\(K\) identical** on every instance.

### Overall means

| | Original TW Classical | Original TW Proposed | Δ | Relaxed TW Classical | Relaxed TW Proposed | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Vehicles | 78.82 | 78.82 | 0 | 65.20 | 65.20 | 0 |
| Distance | 55,119 | 54,782 | −0.61% | 51,088 | 50,596 | −0.96% |
| Duration | 193,319 | 190,177 | −1.63% | 160,768 | 157,645 | −1.94% |
| Station wait | 716.0 | 689.0 | −3.77% | 924.0 | 847.9 | −8.23% |
| Charging time | 19,188 | 12,204 | −36.4% | 20,945 | 14,648 | −30.1% |
| Charging expense | 10,300 | 5,935 | −42.4% | 11,946 | 7,205 | −39.7% |
| Charged energy | 20,511 | 12,747 | −37.9% | 23,049 | 15,434 | −33.0% |
| Customer waiting | 81,629 | 85,836 | +5.2% | 51,146 | 54,887 | +7.3% |
| Runtime (s) | 122.3 | 159.5 | Classical+refine | 95.8 | 144.6 | Classical+refine |

Per-instance wins (Proposed vs Classical): distance, duration, charging time and expense **60/60** on both TW settings; station wait **46/60** Original and **54/60** Relaxed. Fleet never worse.

Family breakdown: `outputs/tables/table2_family.csv`.  
Per-instance: `outputs/tables/per_instance.csv`.  
Writing pack: `outputs/WRITING_PACK.md` and `outputs/HEADLINE_NUMBERS.json`.

The 26 Aug Proposed archive (expense −40.0% / −37.7%) is superseded. Direction and magnitude are the same; the new run is slightly stronger on expense and duration, slightly weaker on total station wait (Original wait wins 49→46) because charging visits rose a little (52.3→54.2).

### How to interpret the large charging-time / expense drop

Most of the charging-time reduction is **charging less energy** (portions retuned; Classical overcharged). That is a valid Stage-2 effect because partial charging is in the model, but it is **not** primarily “smarter geographic station picking,” and it is **not** because Proposed selects faster chargers.

The heterogeneous-station signal is the extra improvement in **cost per unit energy** (−7.3% Original, −9.9% Relaxed) and **waiting per visit** (−7.0% / −8.5%). Charging time per unit energy does **not** generally fall (+2.3% / +4.4%).

Never write: “CARE-FI reduces charging time because it selects faster chargers.”

Write: CARE-FI reduces total charging time mainly by avoiding unnecessary charging, while heterogeneous station selection additionally reduces expected unit charging cost and expected waiting per visit.

### Physical sanity (use as a check, not as an EVRP BKS)

SINTEF Homberger 1000-customer VRPTW BKS mean fleet is about **56.93** vehicles (hierarchical min \(K\) then distance). CARE-FI Original-TW mean \(K=78.82\) is about 22 extra EVs for range, charging time, detours, and timing. The largest relative \(K\) increases are on C2/R2/RC2, where ordinary VRPTW allows very long routes.

Energy: \(Q=600\), \(h=1\). A loose floor on external charging is \(D-600K\). Original Proposed charges \(\approx 12{,}747\) against a floor \(\approx 7{,}492\) (\(\approx 1.70\times\)); Classical was \(\approx 2.62\times\) the same floor. Relaxed Proposed charges \(\approx 15{,}434\) against \(\approx 11{,}476\) (\(\approx 1.35\times\)). Not impossibly low; consistent with correcting overcharge.

\(\Phi\) is a **charging-refinement** objective. It includes station time \(T_{\mathrm{chg}}+W_{\mathrm{station}}\) and does **not** include customer early waiting. Vehicles can therefore arrive earlier at customers and wait there instead. Total route duration still improved on the current 120, but customer waiting rose. Always report \(T_{\mathrm{route}}\), \(W_{\mathrm{customer}}\), and \(W_{\mathrm{station}}\) separately.

Do not compare \(\Phi\) across Constructor / ALNS-style / VNS-style / Classical. Those solvers are not using the same Classical denominators; a self-normalized \(\Phi\approx 3\) is uninformative. Use \(\Phi\) only for Classical → Proposed with frozen \(D^0,T^0,C^0\).

---

## 11. Comparison methods (same semantics, same budget)

Intended matched table, all through `scripts/run_hanif_fps.py`, `--objective classical`, `--budget 180`, seed 0, both TW:

| Method | How | Role in the paper |
|---|---|---|
| Fast constructor | `--disable-intensify` | Weak but feasible fleet-first baseline |
| ALNS-style | `--competitor alns` | Destroy/repair neighbourhood family; **not** a clone of Keskin & Çatay (2016) |
| VNS-style | `--competitor vns_ts` | Relocate / swap / 2-opt; **not** Schneider et al. (2014) VNS+tabu (tabu is unused) |
| Classical (ours) | Stage 1 | Fleet-first with CARE intensification |
| Proposed (ours) | Stage 1 + Stage 2 | Frozen-\(K\) heterogeneous charging refinement |

These are **reimplementations of published neighbourhood families**, not bitwise clones of author codes. None of those papers report this 1000-customer Hanif station set, so comparison is “same algorithmic family on this benchmark,” not a table copied from another paper.

ILS / LNS×GTS / hybrid memetic are catalogued in `src/evaluation/literature_competitors.py` and are **not** reimplemented.

If a competitor archive is not yet complete, the writing agent must not invent ALNS/VNS numbers. Use only CSVs under `outputs/constructor/`, `outputs/alns/`, and `outputs/vns_ts/` after those runs finish, merged by `scripts/compile_paper_results.py`.

---

## 12. Figures — not frozen

The six files currently in `outputs/figures/` are **drafts**. Do not publish them as the final figure set. They mainly contrast Original vs Relaxed TW rather than Classical vs Proposed.

Intended replacements (regenerate only after the corrected Stage 2 is frozen):

1. Method diagram (Stage 1 fleet-first → \(K^*,\mathcal R^*\) → Stage 2 charging).
2. Paired Classical→Proposed deltas on all 120 instances (distance, duration, wait, charge time, expense).
3. Family heatmap.
4. Mechanism decomposition: energy, cost/energy, wait/visit, charge-time/energy.
5. One representative route, Classical vs Proposed charging sequence.
6. Overlay SoC for that same route.

---

## 13. Repository map (for implementers, not for the paper prose)

| Piece | Path |
|---|---|
| Classical runner | `scripts/run_hanif_fps.py` |
| Two-stage experiment | `scripts/run_final_expected.py` |
| \(\Phi\), ranking | `src/evaluation/paper_objective.py` |
| Log-normal | `src/env/lognormal.py` |
| Classical normalisers | `src/evaluation/objective_reference.py` |
| Intensification | `src/search/hanif_fleet_intensify.py` |
| Station pool | `src/search/station_candidates.py` |
| Station refinement | `src/search/station_refine.py` |
| Exact replay | `src/search/route_replay.py` |
| Gate | `src/evaluation/validation_gate.py` |
| Figures | `src/visualization/final_paper_figures.py` |
| Config | `configs/hanif_fps.yaml` |

---

## 14. Sentences the paper may use

Safe:

- The solver first minimises fleet size, then, without adding an EV, refines charging stations and charge amounts.
- Waiting and price enter feasibility and \(\Phi\) through log-normal expectations, not as raw CSV \(\mu\).
- On the current 120, Proposed keeps Classical’s fleet and reduces station waiting, charging time, and charging expense.
- A large share of the charging-time reduction comes from charging less energy; wait per visit and unit price still improve.

Unsafe:

- Proposed finds fewer EVs than Classical.
- Waiting and price drive which stations are generated in the fleet constructor.
- CARE-FI reduces charging time because it selects faster chargers.
- The method is chance-constrained / CVaR-robust / Transformer-MARL.
- We compared against Schneider VNS–TS or Keskin & Çatay ALNS as faithful reproductions.
- CARE-FI beat Keskin & Çatay’s ALNS by 16–29% (those columns are ALNS-style internals).
- Two-phase fleet-first is a novel EVRPTW framework (Wang et al. 2026 already do this).
- Cross-method \(\Phi\approx 3\) is a quality ranking.
- We tested 120 instances under the old `wait_mu`-as-minutes model (that archive is legacy).
- ALNS/VNS numbers, unless the corresponding full CSV exists.

---

## 15. Remaining experiments before freezing Results

Current 120 numbers are usable as a **controlled first full run**, not as the camera-ready package.

Must complete:

1. ~~Regenerate Stage 2 with the balanced pool, \(\Phi\)-burden ranking, and route decomposition (`--skip-classical --full`).~~ **Done (28 Aug).** Scientific PASS. Headline Proposed is `outputs/proposed/` from that run.
2. \(\Phi\) vs \(\Phi_{\mathrm{dur}}\) sensitivity (replace station time by total route duration).
3. Stage-2 ablations: quantity-only, station-only, insert/delete, full.
4. Exact or DP validation of the route-level charging subproblem (gap table).
5. Out-of-sample log-normal simulation (mean, P95, CVaR95, TW/depot feasibility).
6. External Schneider 56-instance FR/PR vs published BKS (`scripts/run_schneider_benchmark.py`). Do **not** treat internal ALNS-style/VNS-style as Schneider/Keskin. Wang E-VRP-HC remains optional (different objective).
7. Stage-1 CARE ablation vs size-only / random elimination.
8. Weights, \(Q\), and Stage-2 budget sensitivity.

CARE-FI’s own fleet/charging numbers can be treated as the current full-run results. Camera-ready **comparison-to-literature** still needs items 6 (and preferably 3). Figures 1–6 still need a Class vs Proposed regeneration.

---

## 16. Development history the paper should not dwell on

An earlier Proposed run **re-solved** the fleet problem with \(\Phi\) in the ranking. \(\Phi\) was implemented correctly, but candidates were still mostly geographically nearest stations, and 1000-customer charging DP was skipped. Result: \(\Phi\approx 3\), wait and price slightly **worse**, fleet worse on several instances. That experiment was discarded and is **not** the paper method.

The two-stage freeze-\(K\) design is the method. The 18-instance gate passed, then the full 60+60 was run.
