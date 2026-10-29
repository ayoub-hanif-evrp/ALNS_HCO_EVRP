# Ready-to-insert results paragraphs

Word file accessibility: see validation report. A revised copy is written when possible as `ALNS-HCO-PAPER_RESULTS_REVISED.docx`. Paste/replace the Results numerical claims with the following (five-seed Hanif unless noted).

## Overall routing (near Table 3)

> Table 3 reports five-seed mean fleet size, distance, duration, and runtime for GI, VNS, SA, ALNS-FR, and ALNS-PR on all 60 instances under each TW regime (GI uses the deterministic construction, seed 0). ALNS variants achieve the smallest mean fleets in both Original and Relaxed TW. Because fleet size is lexicographically primary, distance comparisons are interpreted conditionally when fleets differ.

## Partial recharging (near Figure 2)

> Figure 2 shows paired instance-level percentage changes of ALNS-PR relative to ALNS-FR under equal 30 s budgets. Partial recharging reduces purchased en-route energy, charging visits, station waiting, charging time, and public charging expense on both TW panels (95% bootstrap CIs for these paired percentages exclude zero). Linear constant-rate partial charging changes purchased quantity; it does not, by itself, create nonlinear charging-speed benefits.

## Fixed-route HCO (near Figure 3)

> Figure 3 evaluates Heterogeneous versus Detour HCO on identical Phase-I customer routes. Differences isolate station choice under heterogeneous information without re-routing customers.

## Phase II / HCO (near Figures 4–5)

> Over 600 matched five-seed cases, PhaseII-Heterogeneous matches Phase I on fleet size, distance, and charged energy in **584** cases (97.3%). PhaseII-NoHCO captures Phase-II search without the charging oracle; the NoHCO→Heterogeneous contrast (Figures 4–5) isolates HCO effects and does not show broad dominance. Matching aggregates does not prove identical routes.

## Schneider

> All 56 Schneider large instances were attempted (336 runs). Under the current implementation, 264 runs are feasible and 72 are unsuccessful; primary FR/PR comparisons use the 44 instances feasible for both methods. These instances use homogeneous charging with neutral price/wait placeholders and do not validate heterogeneous pricing or waiting benefits.

## Convergence (supplement)

> On the 288-run two-seed multi-budget panel, longer budgets do not close PR’s relative fleet-size gap versus FR on Original TW (mean gap change 30→300 s for \(K\) has a 95% CI excluding zero). Distance gap changes are inconclusive (CIs include zero). This panel is supplementary to the five-seed 30 s Hanif main study.

## Energy / economics

> Depot replenishment and return residual are reconstructed from the aggregate energy identity when route-level SOC traces are unavailable and are labelled as such. Post-hoc depot tariffs and wages are sensitivity scenarios only; ALNS/HCO do not optimize those totals.
