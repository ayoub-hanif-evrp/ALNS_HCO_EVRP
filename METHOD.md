# CARE-FI method

CARE-FI is a deterministic two-stage EVRPTW solver.

It is **not** “the Hanif method.” Hanif is a 1000-customer benchmark on which CARE-FI is applied. Schneider is a separate published-BKS benchmark.

There is no regret, no lexicographic Stage-1 objective, no random seeds, no ALNS, no VNS, no cold rebuild, and no weighted route score.

## Stage 1 — feasible construction and fleet compression

Construct a feasible plan \(X=\{R_1,\ldots,R_K\}\).

Customers are processed by increasing due time \(b_i\). For each customer \(i\), examine only nearby routes and a few insertion positions. The detour of inserting \(i\) between \(u\) and \(v\) is

\[
\Delta d_{uiv}=d_{ui}+d_{iv}-d_{uv}.
\]

Among the **feasible** insertions, keep the one with minimum \(\Delta d_{uiv}\). If no existing nearby route can receive \(i\), open a new route. A nearby charging station is inserted only when the customer-only plan is battery-infeasible.

Feasibility is a constraint. Distance is the insertion criterion.

### Fleet compression

Consider only the \(q\) smallest routes. For a candidate \(R_r\) with customers \(C_r\), remove \(R_r\) temporarily. For each displaced customer \(i\in C_r\), let \(\mathcal K_i\) be the remaining routes that pass a quick capacity, geographical, and time-window filter.

Reinsert **most-constrained-first**:

\[
i^*=\arg\min_{i\in C_r}|\mathcal K_i|.
\]

This is not regret. For \(i^*\): nearby routes in \(\mathcal K_i\), a few positions, exact EV replay, minimum feasible \(\Delta d\). If any customer fails, restore \(R_r\) immediately. If all succeed, \(K\leftarrow K-1\) and restart. Stop when no candidate can be removed.

The Classical solution is \(X^C\) with fleet \(K^*\).

\[
\text{construct feasible routes}\;\to\;\text{remove one short route}\;\to\;\text{direct reinsertion}\;\to\;\text{repeat}
\]

## Stage 2 — charging refinement

Freeze \(K=K^*\) and every customer sequence. Each existing charging position is one decision: which station and how much to charge, or delete the stop.

\[
\mathcal S^{\mathrm{cand}}=\{S_{\mathrm{current}},\,S_{\mathrm{near}},\,S_{\mathrm{cheap}},\,S_{\mathrm{low-wait}},\,S_{\mathrm{fast}}\}\cup\{\varnothing\}
\]

\[
\alpha\in\{0.25,0.50,0.75,1.00\},\qquad e_s=\alpha(B^{\max}-B_s^{\mathrm{arr}}).
\]

Accept a feasible pair \((s,\alpha)\) only if

\[
\Phi=\frac{D}{D^0}+\frac{T}{T^0}+\frac{C_{\mathrm{ch}}}{C_{\mathrm{ch}}^0}
\]

strictly decreases. \(T\) is **total route duration** (station wait is already inside \(T\)). References \(D^0,T^0,C_{\mathrm{ch}}^0\) come from \(X^C\). Routes are refined independently. Repeat until a full pass does not improve. Then validate the whole solution.

## Validation

Zero hard violations, all customers served, \(K=K^*\), customer sequences unchanged, \(\Phi(X^*)\le\Phi(X^C)\).

## Benchmarks

| | Hanif 1000-customer | Schneider EVRPTW |
|---|---|---|
| Code | `src/hanif/` | `src/schneider/` |
| Stage 1 | feasible insertion + most-constrained reinsertion | lex \((K,D)\) |
| Stage 2 | \(\Phi\) with total duration | not used for BKS |
| Cite frozen 120 | `outputs/hanif/classical/`, `outputs/hanif/proposed/` | `outputs/schneider/pr/` |

The previous Hanif implementation is kept at `src/hanif/current/` until the instance-by-instance \(K\) comparison is accepted.

```powershell
python scripts/run_hanif_carefi.py --representative --tw both --budget 180
python scripts/run_schneider_benchmark.py --mode pr
```

Scratch output is `outputs/_tmp/`. Delete it after you read the numbers.

