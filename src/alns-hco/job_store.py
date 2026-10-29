"""Resumable experiment job store (checkpoint JSON + CSV rebuild)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.common.reporting_io import flatten_solution_row, write_master_csv


def job_id_fr_pr(job: dict) -> str:
    return (
        f"{job['method']}|{job['instance_name']}|{job['tw']}|"
        f"seed{int(job['seed'])}|{float(job['budget_s']):g}s"
    )


def job_id_generic(job: dict) -> str:
    parts = [str(job.get("method")), str(job.get("instance_name") or job.get("instance"))]
    if job.get("tw") is not None:
        parts.append(str(job["tw"]))
    if job.get("seed") is not None:
        parts.append(f"seed{int(job['seed'])}")
    if job.get("budget_s") is not None:
        parts.append(f"{float(job['budget_s']):g}s")
    return "|".join(parts)


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.ckpt = self.root / "checkpoints"
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.runs_csv = self.root / "runs.csv"
        self.status_path = self.root / "status.json"

    def checkpoint_path(self, jid: str) -> Path:
        safe = jid.replace("|", "__").replace("/", "_")
        return self.ckpt / f"{safe}.json"

    def is_done(self, jid: str, *, require_feasible: bool = True) -> bool:
        p = self.checkpoint_path(jid)
        if not p.is_file():
            return False
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        row = rec.get("row") or {}
        if rec.get("error"):
            return False
        if not row:
            return False
        if require_feasible and str(row.get("feasible")).lower() not in {"true", "1"}:
            return False
        return True

    def load_done_ids(self, *, require_feasible: bool = True) -> set[str]:
        out = set()
        for p in self.ckpt.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            jid = rec.get("job_id")
            if jid and self.is_done(jid, require_feasible=require_feasible):
                out.add(jid)
        return out

    def save_success(self, jid: str, job: dict, row: dict) -> None:
        payload = {
            "job_id": jid,
            "job": job,
            "row": flatten_solution_row(row),
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": None,
        }
        self.checkpoint_path(jid).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def save_failure(self, jid: str, job: dict, err: str, tb: str | None = None) -> None:
        payload = {
            "job_id": jid,
            "job": job,
            "row": None,
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": err,
            "traceback": tb,
        }
        self.checkpoint_path(jid).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def collect_rows(self, *, feasible_only: bool = True) -> list[dict]:
        rows = []
        for p in sorted(self.ckpt.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if rec.get("error") or not rec.get("row"):
                continue
            row = dict(rec["row"])
            row["job_id"] = rec.get("job_id")
            is_feas = str(row.get("feasible")).lower() in {"true", "1"}
            if feasible_only and not is_feas:
                continue
            rows.append(row)
        return rows

    def rebuild_csv(self, *, feasible_only: bool = True) -> list[dict]:
        rows = self.collect_rows(feasible_only=feasible_only)
        write_master_csv(self.runs_csv, rows)
        return rows

    def rebuild_csv_all(self) -> list[dict]:
        """Include infeasible rows (for Schneider feasibility tables)."""
        rows = self.collect_rows(feasible_only=False)
        all_path = self.root / "runs_including_infeasible.csv"
        write_master_csv(all_path, rows)
        write_master_csv(self.runs_csv, [r for r in rows if str(r.get("feasible")).lower() in {"true", "1"}])
        return rows

    def write_status(self, status: dict) -> None:
        self.status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
