# Round 1 audit — reviewer revision 2026

**Branch:** `reviewer-revision-2026` (never modify `main` for reviewer work)  
**Git remote:** https://github.com/ayoub-hanif-evrp/ALNS_HCO_EVRP  
**Base:** `origin/main` @ `e8e7ecf` (never modified for reviewer work)  
**Branch:** `reviewer-revision-2026` (Round-1 commits only on this branch)  
**Constraint:** no changes to ALNS operators, HCO decision logic, objectives, or feasibility acceptance.

---

## 1. Architecture and implementation map

### Package aliases

Hyphenated folders are registered as importable packages in `src/__init__.py`:

| Import name | Folder |
|---|---|
| `src.alns_hco` | `src/alns-hco/` |
| `src.hanif` / `src.care_fi` | `src/care-fi/` |

### Phase I / Phase II / baselines

| Method | Role | Entry |
|---|---|---|
| **Phase I (ALNS-PR)** | Shared `X^{PR}` under time budget; lex `(unserved, K, D)`; just-enough charge | `phase1.py`, `alns_pr.py`, `shared_phase1.py` |
| **ALNS-PR (30 s)** | Paper routing baseline = Phase I with full budget | `solver.solve_alns_hco(..., mode="alns_pr")` |
| **ALNS-FR** | Same ALNS loop, **full** recharge | `src/care-fi/benchmarks/alns.py` |
| **FixedRoute-HCO-Detour / Heterogeneous** | Freeze customer sequences; HCO only | `solver` mode `fixed_route_hco` |
| **PhaseII-NoHCO / Plain / Detour / Heterogeneous** | Frozen `K=K*`, distance gate, optional HCO | `phase2.py` + `hco.py` |
| **GI / VNS / SA** | Classical competitors | `src/care-fi/benchmarks/` |

Equal budget (paper): ALNS-PR 30 s vs ALNS-HCO 15 s + 15 s (`src/common/config.py`).

### HCO

Single oracle: `src/alns-hco/hco.py` → `hco()`. Modes:

- **detour** — candidate shortlist by `(detour, station_id)`; non-worsening via geometric score `(D, N_ch, station-id sequence)`.
- **heterogeneous** — top-k union of detour/price/wait/rate, dominance filter, lex `(C_ch, T_wait, T_ch, D)`.

Activity counters: `src/alns-hco/hco_activity.py`.

### Feasibility / metrics calculation

| Layer | File | What |
|---|---|---|
| Authoritative replay | `src/care-fi/replay.py` | Battery, TW, capacity, expected wait/price, `charged_energy` |
| Result rows | `src/alns-hco/station_log.py`, `clean_experiments.py` | Master-CSV fields + station/route logs |
| Aggregation / tables | `src/alns-hco/clean_study.py`, `paper.py` | Paper tables from master CSV |
| Paired stats | `src/common/stats.py` | Wins/ties/losses, Wilcoxon, bootstrap CI |
| Frozen config | `outputs/config/frozen_experiment_config.json` | Experiment freeze |

Scripts: `scripts/run_final_hco_branches.py`, `run_clean_hanif_study.py`, figure generators.

### Datasets

- **Hanif:** `dataset/Hanif/converted_hanif_{original,relaxed}_tw/` — JSON/CSV + `matrices.npz`; station attrs include rate, price, wait, density.
- **Schneider (local):** `dataset/Schneider/evrptw_instances/*.txt` — see §5.

---

## 2. Files modified (Round 1)

| Path | Change |
|---|---|
| `src/common/energy_accounting.py` | **New** — post-hoc energy + search diagnostic helpers |
| `src/common/config.py` | Extended `MASTER_FIELDS`; added `REPORTING_ONLY_FIELDS` |
| `src/alns-hco/station_log.py` | Attach energy fields to route/solution rows (after replay) |
| `src/alns-hco/clean_experiments.py` | Call `_finalize_reporting` after each solve |
| `tests/test_energy_accounting.py` | **New** — balance identity + non-interference checks |
| `outputs/reviewer_revision/REPRODUCE_ORIGINAL.md` | **New** — original reproduction recipe |
| `outputs/reviewer_revision/ROUND1_AUDIT.md` | **New** — this file |
| `.gitignore` | Ignore `**/matrices.npz` (large tensors) |

**Not modified:** `hco.py`, `phase1.py`, `phase2.py`, `alns_pr.py`, destroy/insert operators, replay feasibility equations, hyperparameter values, existing `outputs/tables/` / `outputs/shared_phase1/` numeric artifacts.

---

## 3. Measurements added

Computed **after** the solve from the final routes; never consulted by destroy/repair/HCO/acceptance:

| Field | Definition |
|---|---|
| `propulsion_energy` | Σ arc `energy_matrix` over used routes |
| `enroute_purchased_energy` | Σ charged energy at stations / mid-route depot (same rule as `replay_route`) |
| `residual_battery_at_depot` | Σ SoC at depot return |
| `depot_replenishment_energy` | Σ `(B_max − residual)` |
| `mean_energy_per_charging_visit` | purchased / visit count (or `None`) |
| `initial_battery_energy` | Σ `B_max` over used routes |
| `energy_balance_ok` / `energy_balance_residual` | Checks `B₀ + E_enroute − E_prop = B_final` |
| `alns_iterations`, `alns_phase{1,2}_iterations` | From stage counters when present |
| `hco_calls`, `feasible_hco_calls`, `hco_eval_time_s` | From `HcoActivity` |
| `execution_time_s` | Alias of wall-clock `runtime_s` |

Identity uses existing battery capacity and `energy_matrix` consumption (same as CARE-FI replay).

---

## 4. HCO-Detour audit (documentation only — no fixes)

Reference claims (`src/alns-hco/METHOD.md`, `hco.py` docstring): Detour ranks/selects by `(detour, station_id)` only and never uses \(\bar P, \bar W, r, C_{ch}, T_{wait}, T_{ch}\); non-worsening by `(D, N_ch, station-id sequence)`.

### What matches the claim

| Step | Location | Behavior |
|---|---|---|
| Candidate shortlist | `hco.py` `station_candidates`, **L307–310** | `mode == "detour"`: sort by `(detour, station_id)` only |
| Insertion choice among candidates | `hco.py` `_seed_stations`, **L453–458** | Detour uses `_detour_tie` = `(detour, station_id)` only |
| Final non-worsening | `hco.py` `_geometric_score` **L342–348**, `_accept_score` **L351–354**, `_finalize_vs_input` **L673–692** | Geometric `(D, N_ch, stations)`; no price/wait/rate/expense |
| Heterogeneous contrast | `station_candidates` **L311–329**; ranking via `_lex_charge_score` **L332–339** | Uses price/wait/rate union + lex charge score as documented |

### Inconsistencies (not corrected in Round 1)

1. **Replacement pass mixes total distance into detour candidate ranking**  
   - File: `src/alns-hco/hco.py`  
   - Function: `_select_best_replace`  
   - Lines: **538–545** (replace) and **518–523** (remove)  
   - Detour keys are `(cand.distance, len(stations), detour, station_id)` or `(distance, N_ch, 0)`, not pure `(detour, station_id)`.  
   - Effect: among feasible replacements, **route distance** (and visit count) dominates; insertion-arc detour is only a tertiary key. This diverges from the “rank by `(detour, station_id)` only among feasible candidates” wording, while still avoiding \(\bar P,\bar W,r,C_{ch},T_{wait},T_{ch}\).

2. **Seed / remove keys are not the same tuple shape as `_geometric_score`**  
   - Seed init: **L493–496** → `(seed.distance, N_ch, 0)`  
   - Remove: **L518–519** → `(cand.distance, N_ch, 0)`  
   - `_geometric_score`: **L347–348** → `(distance, N_ch, station-id sequence)`  
   - Final acceptance vs input uses `_geometric_score`, but the internal replace tournament uses the shorter keys above. Station-id sequence is therefore **not** used consistently inside the replace loop.

3. **Comparing replace keys to seed key with trailing `0`**  
   - Because seed `best_key` ends with scalar `0` (**L496**) while a replace candidate ends with `(detour, id)` (**L541**), equal-`(D, N_ch)` replacements with positive detour lose on the third component vs `0`. Improvements in the replace pass therefore require strictly better `(D, N_ch)` first — detour is mainly a tie-break among improving distance candidates, not the primary selection criterion claimed for detour mode.

**Heterogeneous** (for comparison): insertion (**L455–456**) and replace (**L542–543**) both use `_lex_charge_score`; final non-worsening uses the same lex charge score. Internally consistent with its documented criterion.

---

## 5. Schneider compatibility findings

### Format / structure

- Path: `dataset/Schneider/evrptw_instances/` (+ `readme.txt`).
- Per-node columns: `StringID`, `Type` (`d`/`f`/`c`), `x`, `y`, `demand`, `ReadyTime`, `DueDate`, `ServiceTime`.
- Vehicle footer: `Q` (battery), `C` (load), `r` (consumption per distance), `g` (inverse recharge rate), `v` (speed).
- Euclidean distance; travel time = distance / `v`; energy = distance × `r`; **homogeneous** rate `1/g`.
- Instance sizes: 100-customer `*_21` plus smaller `C5`/`C10`/`C15` variants.

### vs Hanif

| Attribute | Hanif | Schneider |
|---|---|---|
| Customers / stations | 1000 / ~333 | 100 / 21 (large set) |
| Station `charging_rate` | Heterogeneous | Homogeneous `1/g` |
| Expected wait \(\bar W\) | `wait_mu` (+ density) | **Absent** |
| Electricity price \(\bar P\) | `price_mu` / expected | **Absent** |
| Density score | Present | **Absent** |
| Converted folder layout | JSON + CSV + npz | Raw `.txt` (converter exists) |

### Support without modifying algorithms (proposal only — not implemented in Round 1)

1. Convert with existing `src/care-fi/data/schneider_txt_convert.py` into the CARE-FI folder schema **without** writing `stations.csv` (loader then disables wait/price; rate falls back to `1/g` via `charging_rate()` in `src/care-fi/model.py`).
2. Run **HCO-Detour** / PhaseII-Detour / geometric baselines as-is (no hetero attrs required).
3. For **HCO-Heterogeneous**, keep the same code path: missing wait/price become 0 and rates collapse to homogeneous — interpret results as a **degraded** heterogeneous oracle on Schneider, or restrict Schneider experiments to Detour / NoHCO / Plain ablations in the paper text.
4. Do **not** invent synthetic prices/waits inside the oracle; any future synthetic attrs must live in the **dataset adapter** only, with explicit sensitivity disclosure.

---

## 6. Test results

```
python -m pytest tests/test_energy_accounting.py tests/test_hco_final_fix.py tests/test_alns_hco.py -q
→ 23 passed in ~7.3 s
```

- Energy balance identity verified on synthetic routes.
- Purchased energy matches `replay_route.charged_energy`.
- Post-hoc accounting does not change HCO/replay objective fields.
- Existing HCO final-fix and ALNS-HCO smoke tests still pass.

No full-benchmark experiments were run in Round 1.

---

## 7. Remaining issues / attention

1. HCO-Detour ranking inconsistencies (§4) — document for reviewers; algorithmic fix deferred.
2. Hanif `matrices.npz` files are gitignored (~2 GB); regenerate locally via Hanif converter before full runs.
3. Local folder initially lacked `.git`; an orphan local history was discarded after discovering the real remote. `reviewer-revision-2026` was rebuilt from `origin/main` (`e8e7ecf`) and Round-1 files re-applied.
4. Schneider full integration deferred to a later round (adapter + experiment harness only).
5. Repo size on GitHub is large (~2.8 GB); use `GIT_LFS_SKIP_SMUDGE=1` and/or partial clone filters for local sync.
