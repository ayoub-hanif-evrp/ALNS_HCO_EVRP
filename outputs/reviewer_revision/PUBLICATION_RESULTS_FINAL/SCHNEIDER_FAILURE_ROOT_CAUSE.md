# Schneider 12-instance failure root cause

## Facts
- Primary 30 s: 12 instances × 3 seeds × 2 methods = 72 unsuccessful attempts (unserved=1).
- Targeted 120 s: 72/72 still unserved=1 → **not a short-timeout artefact**.
- Converter field fidelity: 56/56 OK (no adapter bug found in audited fields).
- Round-4 individual-route probes: customers failing depot–customer–depot often admit a station-assisted
  solo route — so instance-level mathematical infeasibility is **not proven**.

## Missing-customer consistency (this round)
Instances where the **same single customer** is missing in every FR/PR×seed run:
**12 / 12** (see `Schneider_hard_missing_consistency.csv`).

| Instance | Missing customer (all 6 runs) |
|---|---|
| r101_21, r102_21, r103_21, r105_21, r106_21 | C65 |
| r108_21 | C64 |
| rc101_21, rc103_21–rc107_21 | C75 |

The repeated single-customer pattern across methods and seeds indicates a **systematic insertion/construction barrier** for that customer under the current search stack, not random timeout noise.

## Classification
**Construction / search / initialization limitation under the current solver stack**, not a verified
converter parameter error, and not a proof of official-instance infeasibility under every distance-rounding
convention.

Further multi-hour ALNS runs are unlikely to be informative given the uniform 120 s outcome; algorithm
changes to force feasibility are out of scope for this revision.
