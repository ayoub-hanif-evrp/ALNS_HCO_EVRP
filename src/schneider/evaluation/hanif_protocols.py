"""Named Hanif evaluation protocols used in paper metadata.

EXTENDED is the optimization / reporting protocol. NO_WAIT_REPLAY replays the
same saved routes and charging labels with station waits disabled.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

PROTOCOL_EXTENDED = {
    "name": "EXTENDED",
    "alias": "HANIF_EXTENDED_DETERMINISTIC",
    "battery_Q": 600,
    "energy_h": 1.0,
    "num_stations": 333,
    "charging_rate_heterogeneous": True,
    "station_wait": "deterministic_expected_lognormal_wait",
    "wait_sigma_sampled": False,
    "partial_charging": True,
    "charging_price_in_objective": False,
    "expected_wait_in_objective": False,
    "density_score_in_search": False,
}

PROTOCOL_PROPOSED_EXTENDED = {
    **PROTOCOL_EXTENDED,
    "name": "EXTENDED",
    "alias": "HANIF_EXTENDED_DETERMINISTIC_PROPOSED",
    "charging_price_in_objective": True,
    "expected_wait_in_objective": True,
    "objective": "Phi = D/D_class + Tstation/Tstation_class + Ccharge/Ccharge_class",
}

PROTOCOL_NO_WAIT_REPLAY = {
    "name": "NO_WAIT_REPLAY",
    "parent": "EXTENDED",
    "battery_Q": 600,
    "energy_h": 1.0,
    "num_stations": 333,
    "charging_rate_heterogeneous": True,
    "station_wait": "disabled_wait_mu_zero",
    "wait_sigma_sampled": False,
    "partial_charging": True,
    "charging_price_in_objective": False,
    "density_score_in_search": False,
    "preserves_customer_and_charging_sequences": True,
    "reoptimizes": False,
}

PROTOCOL_ALIASES = {
    "EXTENDED": "EXTENDED",
    "HANIF_EXTENDED_DETERMINISTIC": "EXTENDED",
    "HANIF_EXTENDED_DETERMINISTIC_PROPOSED": "EXTENDED",
    "NO_WAIT_REPLAY": "NO_WAIT_REPLAY",
}


def evaluation_protocol(config: dict | None = None) -> dict[str, Any]:
    """EXTENDED metadata overlay: Proposed Φ includes charging price and station wait."""
    from src.schneider.evaluation.paper_objective import MODE_CLASSICAL, objective_spec

    spec = objective_spec(config)
    if spec["mode"] == MODE_CLASSICAL:
        return dict(PROTOCOL_EXTENDED)
    return dict(PROTOCOL_PROPOSED_EXTENDED)

FROZEN_HEADLINE = {
    "path": "outputs/hanif/",
    "classical": "outputs/hanif/classical",
    "proposed": "outputs/hanif/proposed",
    "method": "Classical fleet search then heterogeneous station refinement",
    "role": "paper_headline_corrected_wait",
    "A_mean_k": 78.82,
    "B_mean_k": 65.20,
    "n": 60,
}

CARE_FI_STATUS = {
    "role": "abandoned_pilot",
    "archive": "removed",
    "not_headline": True,
}


def canonical_protocol_name(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("name") or value.get("alias")
    key = str(value).strip()
    return PROTOCOL_ALIASES.get(key, key)


def protocols_compatible(left: Any, right: Any) -> bool:
    a, b = canonical_protocol_name(left), canonical_protocol_name(right)
    if a is None or b is None:
        return True
    return a == b


def instance_without_station_wait(instance):
    """Clone an instance with expected station wait forced to zero (no re-routing)."""
    zeros = np.zeros(int(instance.num_nodes), dtype=float)
    return replace(
        instance,
        station_wait_mu=zeros,
        station_wait_sigma=zeros,
        station_wait_expected=zeros,
    )


def completeness_report(
    *,
    expected_names: Iterable[str],
    got_names: Iterable[str],
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = [str(n) for n in expected_names]
    got = [str(n) for n in got_names]
    got_set = set(got)
    missing = [n for n in expected if n not in got_set]
    extra = [n for n in got if n not in set(expected)]
    failures = list(failures or [])
    complete = not missing and not failures
    return {
        "expected_n": len(expected),
        "n": len(got),
        "complete": complete,
        "missing_instances": missing,
        "extra_instances": extra,
        "failed_instances": failures,
        "cite_means": complete,
    }


def means_or_incomplete(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Return a citeable mean only when the caller already marked completeness."""
    values = [float(r[key]) for r in rows]
    partial = sum(values) / max(len(values), 1) if values else None
    return {
        "n_rows": len(values),
        "partial_mean": partial,
    }
