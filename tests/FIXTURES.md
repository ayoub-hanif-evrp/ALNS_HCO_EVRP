# Dataset / baseline fixtures for tests

- Unit tests that need Schneider instances skip with a clear reason when
  `dataset/Schneider/evrptw_instances/...` is absent.
- Frozen baselines:
  - `outputs/baselines/full9_20260731/` — Expert / BC / Old RL-best (immutable)
  - `outputs/baselines/expert_repair_alns_20260801/` — deterministic Expert+Repair+ALNS
    full-9 milestone (**not** Transformer-guided)
- Verify: `python scripts/verify_tgfps_hashes.py`
