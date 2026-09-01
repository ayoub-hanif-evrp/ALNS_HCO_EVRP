# ALNS-HCO method (paper)

ALNS-HCO is Adaptive Large Neighborhood Search with a Heterogeneous Charging
Oracle. Hanif is the 1000-customer **benchmark**. ALNS-HCO is the **method**.

This file describes the **active** execution path in `src/alns-hco/`.

## Phase I — ALNS-PR

GI constructs a feasible start. ALNS then destroy/repairs for `phase1_time_limit_s`
(15 s in the equal-budget experiments). Repair uses **just-enough partial
recharge**, not full recharge.

The lexicographic score is (unserved, \(K\), distance). The incumbent fleet is

\[
K^*.
\]

Phase I **is** the ALNS-PR baseline when it is given the same 30 s total budget.
Inside ALNS-HCO, Phase I uses 15 s so that Phase II also receives 15 s.

## Phase II — frozen fleet, charging-aware ALNS + HCO

Freeze

\[
K = K^*.
\]

Phase II never accepts a candidate with a different number of vehicles.
Customer-to-vehicle assignment and customer order may change.

Repair inserts removed customers with just-enough charge and then calls **HCO
on the modified route**. After a complete candidate is assembled, HCO is applied
to every route and the SA score uses the HCO plan (distance, then charging
expense).

## One HCO

There is a single function `hco` (`src/alns-hco/hco.py`). It is a deterministic
route-level charging **heuristic**: insert a station if the sequence is
battery-infeasible, replace or drop redundant stops, and set just-enough
portions. Station candidates mix proximity, rate \(r_s\), expected wait
\(\bar W_s\), and expected price \(\bar P_s\).

It is not an exact algorithm and does not claim global optimality.

Names `fast_hco` / `exact_hco` in `oracle.py` are aliases of `hco` for old tests.

## Equal budget

The paper comparison is

\[
\text{ALNS-PR (30 s)}\quad\text{vs}\quad\text{ALNS-HCO (15 s + 15 s)}.
\]

ALNS-FR is the same ALNS loop with full recharge, also 30 s.
ALNS-PR+HCO freezes Phase-I customer sequences and applies HCO only (ablation).

## Seeds

Stochastic methods use seeds \(\{0,1,2,3,4\}\) (\(n=5\)).
