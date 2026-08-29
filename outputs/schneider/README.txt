CARE-FI vs published Schneider EVRPTW BKS
==========================================
Instances: 56 large (100 customers, 21 stations). Source: Schneider et al. 2014
(Mendeley 10.17632/h3mrm5dhxw.1). BKS: Erdelic & Caric, Energies 2022, 15:285
(consolidation of Schneider, Goeke, Hiermann, Keskin, Schiffer).

Lexicographic gap: Delta_K = K_CARE - K_BKS. Distance gap only when Delta_K = 0.
FR = full recharge (portion 1.0). PR = discrete portions 0.25/0.50/0.75/1.0
(not continuous Keskin recharge). No station wait/price. Seed 0, 180 s, 8 workers.

This is an external literature benchmark. It does NOT use the internal
ALNS-style / VNS-style Hanif codes.

FR (full recharge)
  feasible 54/56 (infeasible: r102_21, r103_21)
  fleet wins 0, ties 2 (c203_21, r203_21), losses 52
  mean K CARE 11.21 vs BKS 7.86, mean Delta_K +3.31 (feasible only)
  mean runtime 163 s

PR (partial recharge portions)
  feasible 54/56 (infeasible: r102_21, r103_21)
  fleet wins 0, ties 2 (c203_21, r203_21), losses 52
  mean K CARE 11.38 vs BKS 7.61, mean Delta_K +3.70 (feasible only)
  mean runtime 176 s

Paper sentence that is allowed:
  On the 56 Schneider instances, CARE-FI found feasible solutions for 54/56
  (FR and PR). Relative to published BKS it tied the fleet on two instances
  and used more vehicles on the rest (mean +3.3 EVs FR, +3.7 PR). It did not
  improve any published fleet BKS in this first 180 s single-seed run.

Paper sentence that is not allowed:
  CARE-FI outperformed Schneider / Keskin / Schiffer on EVRPTW.
