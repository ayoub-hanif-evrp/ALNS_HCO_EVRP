"""Read-only dataset integrity checks at training startup."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.data.evrptw_dataset import EVRPTWDataset
from src.data.instance_loader import load_capacity_from_config
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def log_dataset_immutability(config: dict, train_names: list[str] | None = None) -> dict:
    """Log checksums and TW stats; never writes instance files."""
    ds_cfg = config.get("dataset") or {}
    root = Path(ds_cfg["path"])
    load_cap = load_capacity_from_config(config)
    ds = EVRPTWDataset(root, load_capacity=load_cap)

    names = train_names or ds.names
    meta: dict = {"instances": {}, "dataset_root": str(root)}
    all_ready, all_due = [], []

    for name in names:
        if name not in ds.names:
            continue
        idx = ds.names.index(name)
        inst = ds[idx]
        folder = root / name
        customers_csv = folder / "customers.csv"
        digest = _file_hash(customers_csv) if customers_csv.exists() else "missing"
        ready = [float(inst.nodes[c].ready_time) for c in inst.customer_indices]
        due = [float(inst.nodes[c].due_time) for c in inst.customer_indices]
        all_ready.extend(ready)
        all_due.extend(due)
        meta["instances"][name] = {
            "customers_hash": digest,
            "num_customers": inst.num_customers,
            "num_stations": len(inst.charging_station_indices),
            "min_ready": min(ready) if ready else 0.0,
            "max_due": max(due) if due else 0.0,
        }

    LOGGER.info(
        "Dataset immutability: %d train instances, read-only, no TW modification. "
        "Global ready [%.1f, %.1f] due [%.1f, %.1f]",
        len(meta["instances"]),
        min(all_ready) if all_ready else 0.0,
        max(all_ready) if all_ready else 0.0,
        min(all_due) if all_due else 0.0,
        max(all_due) if all_due else 0.0,
    )
    return meta
