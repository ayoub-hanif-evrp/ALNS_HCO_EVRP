# Milestone report — scientific rebuild (P0–P5 foundation)

## Phase 0 — FixedFleet baseline preserved
- **files:** `outputs/baselines/fixed_fleet_expert_repair_alns/`
- **method name:** `FixedFleet-Expert+Repair+ALNS`
- **explicit note:** fleet minimization was **NOT** performed (`m=40` fixed upper bound)
- **command:** copy from prior Expert+Repair+ALNS holdout + `manifest.json`
- **artifact:** `manifest.json`, `README.md`, `recomputed_table.json`

## P0 — Reproducibility
- **tests added:** `tests/test_deterministic_core.py`
- **script:** `scripts/recompute_final_tables.py`
- **fixtures note:** `fixtures/README.md`, `tests/FIXTURES.md`
- **command:** `pytest -q` → **43 passed**
- **command:** `python scripts/recompute_final_tables.py`

## P1–P2 — BenchmarkScore + fleet minimization API
- **files:** `src/search/score.py` (`BenchmarkScore` lex order: hard→unserved→vehicles→distance→charging→duration→J)
- **files:** `src/search/fleet_search.py` (`solve_with_fleet_search`, warm-start, diversified retries, attempt traces)
- **config:** `configs/train_tgfps.yaml` fleet_search retries/seeds; neural BC guide **disabled**
- **hybrid:** `solve_expert_repair_alns(..., m=None)` uses fleet search; explicit `m` = FixedFleet ablation only

## P3 — Real portfolio
- **files:** `src/search/portfolio.py` — 5 variants × 4 seeds by default; elite pool by score + edge/assignment/charging diversity

## P4 — Charging DP
- **files:** `src/search/charging_dp.py` — customer sequence + optimized partial-charge plan (exact replay validated)

## P5–P9 (partial)
- **incremental prechecks:** `src/search/incremental.py`
- **critical rescue:** `src/search/critical_rescue.py` (generic, no filename branches)
- **route pool + set partitioning:** `src/search/route_pool.py`, `src/search/set_partitioning.py`
- **Deterministic-FPS entrypoint:** `src/search/deterministic_fps.py`

## Not done yet (next)
- Full ALNS/VNS operator expansion (Or-opt, 2-opt*, cross-exchange, …)
- Freeze Deterministic-FPS after equal-budget validation on 6 instances
- P11+ Transformer **search-move** guide (not node BC)

## Honest status
- FixedFleet 8/9 result remains the preserved baseline; it is **not** minimized-fleet.
- Neural construction guide remains **rejected**; BC checkpoint must not guide search.
- Competitive Deterministic-FPS numbers on full-9 are **not** claimed until fleet-minimized eval is run and frozen.
