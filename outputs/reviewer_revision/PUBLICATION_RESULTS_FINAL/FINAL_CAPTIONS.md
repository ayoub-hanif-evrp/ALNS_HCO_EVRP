# Final captions and numbering

Sign convention for paired % figures: mean of instance-level \(100(B-A)/A\) after averaging seeds within instance; **negative = treatment lower than baseline** (improvement for lower-is-better metrics).

## Main paper

**Table 3.** Overall routing performance of the baseline methods (GI, VNS, SA, ALNS-FR, ALNS-PR) under Original and Relaxed TW. Means are five-seed instance averages (GI: deterministic seed 0). Fleet size is primary.  
File: `FINAL_TABLES/T3_overall_routing_performance.csv`

**Figure 2.** Effect of Partial Recharging on Routing and Charging Performance. Paired ALNS-FR → ALNS-PR percentage changes with 95% bootstrap CIs.  
Files: `FINAL_FIGURES/Figure2_partial_recharging_FR_vs_PR.{png,pdf}`

**Figure 3.** Fixed-Route Effect of Heterogeneous Charging Information. FixedRoute-HCO-Detour → FixedRoute-HCO-Heterogeneous.  
Files: `FINAL_FIGURES/Figure3_fixed_route_Detour_vs_Het.{png,pdf}`

**Table 4 (contribution).** Improved / unchanged / worsened counts for Phase I ↔ NoHCO ↔ Heterogeneous over 600 five-seed matched cases. Aggregate identity does not imply route identity.  
File: `FINAL_TABLES/T4_HCO_contribution_counts_seeds0to4.csv`

**Figure 4.** Effect of HCO Relative to the Matched Phase-II NoHCO Search.  
Files: `FINAL_FIGURES/Figure4_PhaseII_NoHCO_vs_Heterogeneous.{png,pdf}`

**Figure 5.** Family-Level Effect of HCO in the Matched Phase-II Search (charging expense %; means only).  
Files: `FINAL_FIGURES/Figure5_family_PhaseII_NoHCO_vs_Het_Cch.{png,pdf}`

**Table 5.** Mean paired change in charging expense by instance family (%).  
File: `FINAL_TABLES/T5_family_HCO_effects.csv` (filter `comparison=phase2_nohco_hco` and/or `fixed_route`)

## Supplementary / extended

**Table 6.** Energy accounting (propulsion, en-route, reconstructed depot/residual, visits, waiting, time, public cost).  
**Table 7.** Schneider feasibility (336) and conditional FR/PR (44).  
**Table 8.** FR/PR multi-budget convergence (288) and 30→300 gap changes.  
**Table 9.** Post-hoc depot electricity and wage sensitivity (not optimized).

**Figure S (convergence).** `FigS_FR_PR_convergence_*.png` — two-seed supplementary panel; do not mix into five-seed primary claims.

**Schneider FR168 (results package).** Externally verified full-recharge ALNS-FR routes (168 attempts). Instance-level table uses best-of-three valid seeds (K then distance), not averaging. Files: `Schneider_FR168_*.csv`, `Schneider_BKS_FR_provenance_56.csv`.

**Author warning:** Do not copy three-seed HCO identity **350/360** from historical response drafts; five-seed identity is **584/600**.
