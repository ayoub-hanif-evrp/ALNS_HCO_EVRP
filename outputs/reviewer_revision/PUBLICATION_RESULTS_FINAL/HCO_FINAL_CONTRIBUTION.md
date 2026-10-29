# HCO final contribution analysis (five-seed)

## Headline (verified, not rerun)
- Phase I → PhaseII-Heterogeneous identical on K, distance, charged energy: **584/600**.
- Route identity is **not** claimed from aggregate matches.
- NoHCO → Het charging expense (run-level, both TW): see `HCO_FINAL_runlevel_counts_by_tw.csv`
  (overall previously verified 16 / 501 / 83).

## Why a favorable mean can coexist with many worsenings
Instance-level means can be pulled by a minority of large improvements (notably **C1 / Original TW**),
while most matched runs are unchanged and a non-trivial minority worsen. Medians and leave-one-family-out
tables (`HCO_family_Cch_leave_one_out.csv`) make this visible.

## Contrasts reported
PhaseI↔NoHCO, PhaseI↔Het, NoHCO↔Het, PhaseII-Detour↔Het, FixedRoute-Detour↔Het — run-level counts by TW
and instance-level paired % with bootstrap CIs (`HCO_FINAL_instance_paired.csv`).

## Hyperparameters
- `candidates_per_criterion` ∈ {2,5,8} on the development subset: **identical** solution metrics (prior 72-run study).
  **Five candidates is a fixed design choice, not a proven optimum.**
- Existing ε∈{0,0.01,0.02,0.05} and 10/20–15/15–20/10 panels produced ties (ΔCch=0) on the development subset.
- Frozen primary settings were **not** retuned after seeing full-benchmark results.
