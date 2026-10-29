# Final scientific results report

**Branch:** `reviewer-revision-2026`  
**Protocols:** Hanif **5 seeds** (0–4); Schneider **3 seeds**; FR/PR convergence **288** (seeds 0–1).  
**No algorithm changes.** Numbers recomputed from archived solves.

## Authoritative sources

| Protocol | Source | N |
|---|---|---|
| Hanif main | `outputs/raw/solution_results.csv` | 6720 (incl. GI/VNS/SA); 7-method panel 4200 |
| Schneider 30 s | `outputs/reviewer_revision/schneider/` | 336 attempts |
| FR/PR convergence | `fr_pr_convergence_primary288/` | 288 |

Original vs Relaxed TW are related variants analyzed separately. Seeds averaged within instance before instance-level paired CIs. Fleet size \(K\) is lexicographically primary.

## Key findings

### Routing baselines (Table 3 / `T3_overall_routing_performance.csv`)
ALNS-FR and ALNS-PR attain the smallest mean fleets among GI/VNS/SA/ALNS on both TW regimes (five-seed instance means). Values match the archived publication overall table (validation: 144 matched metric cells, 0 mismatches).

### Full vs partial recharging (Figure 2)
Under equal 30 s budgets, ALNS-PR reduces en-route charged energy, charging visits, waiting, and public charging expense vs ALNS-FR on both TW panels (paired % CIs exclude 0 for these charging metrics). Distance and fleet effects must be read lexicographically: PR is not claimed superior when it uses more vehicles.

### Fixed-route HCO (Figure 3)
On identical Phase-I customer routes, Heterogeneous vs Detour yields small paired % changes in expense/waiting (see `T_fixed_route_Detour_vs_Heterogeneous.csv`).

### Phase II / HCO contribution (Figures 4–5; five-seed audit)
- Phase I → PhaseII-Heterogeneous: **584/600** matched cases unchanged on \(K\), distance, and charged energy (97.3%).
- Phase I → PhaseII-NoHCO captures ordinary search improvement without HCO.
- PhaseII-NoHCO → Heterogeneous isolates the HCO channel (mostly small / mixed; not broad dominance).
- **Route identity is not claimed** from aggregate matches.

### Schneider (Table 7)
336 attempts; **264 feasible**, **72 unsuccessful**; FR/PR performance conditional on **44** both-feasible instances. Neutral charging assumptions (no fictitious price/wait heterogeneity). Failures classified as construction/search limits, not proven mathematical infeasibility.

### Convergence (Table 8; supplementary figures)
Primary 288 only. On Original TW, mean change in (PR−FR) fleet gap from 30→300 s is positive with CI excluding 0 (PR’s relative \(K\) disadvantage does not close). Distance gap-change CIs include 0. Equal wall-clock ≠ equal ALNS iteration throughput.

### Energy and economics
Depot replenishment and residual battery are **reconstructed** via the aggregate identity from distance×rate and observed charged energy (`T6_*`). Public charging expense is observed. Depot-tariff and wage totals in `T9_*` are **post-hoc** and not optimized.

## Limitations
1. Aggregate PhaseII≡PhaseI match ≠ proven route identity.  
2. Schneider BKS / rounding sensitivity not claimed.  
3. Family Figure 5 shows means only (no fabricated family CIs).  
4. Economic scenarios are hypothetical.  
5. Convergence panel is a 12-instance subset (supplementary).
