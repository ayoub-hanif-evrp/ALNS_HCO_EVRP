"""Customer-route checksums for shared Phase-I identity checks."""
from __future__ import annotations

import hashlib
import json

from src.hanif.replay import customer_sequence


def customer_routes_payload(instance, routes: list) -> list[list[int]]:
    """Ordered list of customer sequences (one per nonempty route)."""
    out = []
    for item in routes:
        nodes = item[0] if isinstance(item, (list, tuple)) and item and not isinstance(item[0], int) else item
        seq = list(customer_sequence(instance, list(nodes)))
        if seq:
            out.append(seq)
    return out


def customer_checksum(instance, routes: list) -> str:
    """SHA-256 of the ordered customer sequences. Stations/labels ignored."""
    payload = customer_routes_payload(instance, routes)
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_same_checksum(instance, routes_a: list, routes_b: list, *, context: str = "") -> str:
    ca = customer_checksum(instance, routes_a)
    cb = customer_checksum(instance, routes_b)
    if ca != cb:
        raise AssertionError(f"customer checksum mismatch {context}: {ca} != {cb}")
    return ca
