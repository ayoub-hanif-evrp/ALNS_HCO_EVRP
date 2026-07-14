# Technical report — Exact EVRPTW correctness + performance redesign
# Date: 2026-07-14

## Summary

Correctness bugs in the exact Schneider EVRPTW solver were fixed, unsafe
pruning was disabled by default, bounded multi-hop recharge reachability was
added, and mask computation was restructured for caching/vectorization.
Phase 0 (14/14) passes. Eight standalone correctness checks pass.
Cold `rc102_21` decision analysis ≈ **71 ms** (was ~3.5–6.7 s depending on path).

Full cyclic training was **not** started — Stage 4 overfit gate is next.

## Files changed (main)

| Area | Files |
|------|--------|
| Direct-depot feature | `src/models/dynamic_candidate.py` |
| Critic padding masks | `src/models/critic.py` |
| Per-EV served counter | `src/env/ev_state.py`, `src/env/evrptw_env.py` |
| Auto-close invalidation | `src/env/evrptw_env.py` |
| TW vs unserviceability | `src/env/serviceability.py`, `src/env/action_space.py` (GlobalServiceability) |
| Charging-chain search | `src/env/recharge_search.py` (new) |
| Safe pruning | `src/env/mask_layers.py`, `src/env/action_space.py` |
| Decision analysis | `src/env/decision_analysis.py` (new) |
| Instance hot arrays | `src/data/instance_loader.py` |
| Diagnostics/checkpoint | `src/training/ac_train.py`, `src/training/train_rl.py`, `src/training/config_validation.py` |
| Config | `configs/train.yaml`, `configs/train_schneider_RC21_exact_optimized.yaml` (new) |
| Tests | `tests/test_correctness_fixes.py`, `tests/performance/test_mask_performance.py` |

## Correctness bugs fixed

1. **Direct-depot battery double subtraction** — dynamic features required
   `battery_after >= 2 × energy_to_depot`. Now matches `safe_depot_return_from`
   (single subtraction + independent time check).
2. **Batched critic** — replaced zeroing keys with real `key_padding_mask`;
   empty selections return true zeros (matches single-state).
3. **`num_customers_served`** — was `len(route_nodes)-1` (stations/depot inflated).
   Now an exact integer incremented only on newly served customers.
4. **TW expiration conflation** — unserviceable customers were labeled “expired”.
   Separate lists: TW-expired, battery, capacity, no-safe-continuation,
   general unserviceable. Newly-expired penalty uses TW-only counts.
5. **Charging chains** — bounded label-setting search over depot+stations
   (`max_station_hops: 3`, dominance pruning). Used in serviceability.
6. **Unproven station pruning** — `prune_station_when_customer_available: false`.
   Proactive charging kept when customers are already feasible.
7. **Auto-close at depot** — now bumps `state_version` and clears mask /
   serviceability / analysis / recharge caches even when already at depot.
8. **Terminal penalty schedule** — disabled (`enabled: false`); stationary target.

## Complexity / caches / vectorization

**Old hot path (conceptual):**
`EVs × customers × stations × portions × customers` nested Python loops,
repeated per subsystem (mask, usefulness, serviceability, features).

**New structure:**
- `analyze_decision_state` (reference + optimized) computes once per
  `(state_version, ev_id)` and is reused.
- Vectorized basic customer prefilter (`travel`, `battery`, `capacity`, `TW`)
  before expensive safe-continuation.
- Instance arrays cached at load (`demand/ready/due/service`, depot-return energy).
- Feasibility scratch + action-mask + decision-analysis + global-serviceability
  caches keyed by state version; invalidated on every mutation including
  at-depot auto-close.
- Recharge simulation results memoized inside scratch (`charge_portions`,
  `charge_analysis`).

## Timing (rc102_21, cold decision, development machine)

| Metric | Before (recent) | After |
|--------|-----------------|-------|
| Cold mask / analysis | ~0.8–3.5 s (earlier) / ~6.7 s (with heavy usefulness) | **~71 ms** |
| Speedup vs heavy path | — | **~90×** vs 6.7 s; **~10–50×** vs prior cold |
| Mask parity ref↔opt | — | **identical** (node/charge/depot/serviceability) |
| Dynamic features | — | Match except optional `enabled_after` when details disabled |

Absolute target `< 20 ms` is not yet met on this hardware for full exact
analysis including global serviceability; residual cost is still in
`customer_feasible` / hard-layer charge continuation. Further Numba/Cython
vectorization of safe-continuation remains the main bottleneck.

## Tests

Standalone script (pytest segfaults on this Windows/pyarrow stack):
- direct-depot feature vs mask
- per-EV served counter
- auto-close cache invalidation
- reference/optimized analysis parity
- TW category separation
- recharge-chain search runs
- batched vs single critic
- safe pruning keeps strategic stations

Phase 0: **14/14 PASS**

## Remaining bottlenecks

1. Hard-layer `customer_feasible` + nested `feasible_charge_portions` still
   Python-loop heavy for survivors of the vectorized prefilter.
2. Global serviceability Phase 2 still invokes recharge-chain search per
   pending customer×EV when needed.
3. Dynamic features still built on CPU NumPy then copied to GPU per decision
   (static embeddings already cached once per rollout).
4. Full Stage 4–7 training gates not yet executed.

## Checkpoint compatibility

Architecture/critic attention masking and EV served counters change critic
inputs. **Previous experiment checkpoints are not fully compatible** with the
corrected critic/feature semantics — treat old results as approximate for
feasibility trends, not final scientific numbers.

## Previous experiment trustworthiness

| Aspect | Trust |
|--------|-------|
| Exact TW / capacity / battery hard masks | Mostly ok |
| Dynamic `safe_direct` features | **Not trustworthy** (double energy) |
| Critic values (batched) | **Not trustworthy** (padding dilution) |
| Per-EV fleet balance / served counts | **Not trustworthy** (route-length proxy) |
| Newly-expired penalties | **Overstated** (battery as “expired”) |
| Station action rarity under old pruning | **Biased** (proactive charging pruned) |

## Official config

`configs/train_schneider_RC21_exact_optimized.yaml`

## Recommended next stages (do not skip)

```powershell
# Stage 3 — already partially measured; optional:
python -m pytest tests/performance -q   # may segfault on this machine; use scripts

# Stage 4 — one-instance overfit (≥500 updates)
python main_train.py --config configs/train_schneider_RC21_exact_optimized.yaml --skip-overfit --device cuda
# with training.debug_overfit_one_instance: true and debug_overfit_updates: 500

# Stage 5 — 5 cycles × 13 RC instances (only after Stage 4 looks healthy)
# Stage 6–7 — 20 then 100 cycles
# Mixed C+R+RC — only after RC feasibility/stability demonstrated
```

## Mixed-class curriculum (deferred)

Balanced C / R / RC sampling and per-class holdouts are required **after**
RC `_21` overfit + short cyclic stability, not before.
