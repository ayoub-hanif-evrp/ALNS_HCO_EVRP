# PhaseII-Heterogeneous vs Phase I — diagnostic

- Compared keys: **12**
- Unchanged (K, distance, charged_energy): **12** (100.0%)
- Configured `phase2_distance_tolerance` (eps): **0.0**

## Evidence from Round-2 absolute CSV

- PhaseII-Heterogeneous rows retain `hco_mode=heterogeneous` and non-zero `hco_calls`.
- PhaseII-NoHCO can slightly change distance (ordinary ALNS move) while Het stays on Phase I,
  consistent with HCO-guided repair failing the eps=0 distance gate more often.
- Acceptance stats are not always exported as `accepted=0` columns on the absolute CSV;
  identical Phase-I metrics + positive HCO evals imply accepted improving moves = 0.

## Conclusion (no algorithm change)

On the preliminary absolute-results slice, PhaseII-Heterogeneous often returns the Phase-I routes unchanged. Recorded rows still show positive hco_calls / feasible_hco_calls and phase2 iterations, so HCO is evaluated. Acceptance requires phase2_admissible (K<=K*, D<=D_PR*(1+eps) with eps=0) plus lexicographic improvement. With a tight distance freeze, many destroy/repair moves are rejected on distance even when HCO finds feasible recharges. This is a documented behavioural finding — the algorithm was not altered in Round 3.
