# Fixtures

- `fixtures/` — reserved for tiny synthetic instances checked into git.
- External Schneider instances under `dataset/Schneider/` are optional for CI;
  tests call `pytest.skip("external Schneider fixture ... absent")` when missing.
- Frozen baselines under `outputs/baselines/` are scientific artifacts; hash-verify with
  `python scripts/verify_tgfps_hashes.py` and `python scripts/recompute_final_tables.py`.
