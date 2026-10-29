"""JobStore resume semantics for feasible vs attempted."""
from __future__ import annotations

from src.alns_hco.job_store import JobStore


def test_infeasible_not_done_but_attempted(tmp_path):
    store = JobStore(tmp_path)
    jid = "ALNS-FR|x|seed0|30s"
    store.save_success(jid, {"method": "ALNS-FR"}, {"feasible": False, "unserved": 1, "distance": 1.0})
    assert store.is_done(jid, require_feasible=True) is False
    assert store.is_done(jid, require_feasible=False) is True
    assert jid not in store.load_done_ids(require_feasible=True)
    assert jid in store.load_done_ids(require_feasible=False)
    all_rows = store.rebuild_csv_all()
    assert len(all_rows) == 1
    assert store.collect_rows(feasible_only=True) == []
