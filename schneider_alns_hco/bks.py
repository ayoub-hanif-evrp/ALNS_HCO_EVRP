"""Published EVRPTW-PR best-known solutions for the 56 large Schneider instances.

Transcribed from Erdelić & Carić (2022), Energies 15:285, Table 3. That table
already consolidates Schneider 2014, Goeke & Schneider, Hiermann et al.,
Keskin & Çatay (2016), and Schiffer & Walther.

Literature ranking: feasible, then min K, then min distance.
The BKS used here is the lexicographic min of (literature column, Erdelić ALNS).
"""
from __future__ import annotations

from typing import Any

# instance, lit_source, lit_k, lit_d, erdelic_k, erdelic_d
_PR: list[tuple[str, str, int, float, int, float]] = [
    ("c101_21", "SCHI", 12, 1043.38, 12, 1043.38),
    ("c102_21", "SCHI", 11, 1029.44, 10, 1058.67),
    ("c103_21", "SCHI", 10, 971.86, 10, 971.19),
    ("c104_21", "SCHI", 10, 884.38, 10, 884.38),
    ("c105_21", "KESK", 11, 1037.78, 10, 1064.66),
    ("c106_21", "SCHI", 11, 1010.56, 10, 1061.61),
    ("c107_21", "SCHI", 10, 1010.91, 10, 1046.50),
    ("c108_21", "SCHI", 10, 1031.85, 10, 1022.93),
    ("c109_21", "SCHI", 10, 940.38, 10, 940.38),
    ("c201_21", "KESK", 4, 629.95, 4, 629.95),
    ("c202_21", "KESK", 4, 629.95, 4, 629.95),
    ("c203_21", "KESK", 4, 629.95, 4, 629.95),
    ("c204_21", "SCHI", 4, 628.91, 4, 628.91),
    ("c205_21", "KESK", 4, 629.95, 4, 629.95),
    ("c206_21", "KESK", 4, 629.95, 4, 629.95),
    ("c207_21", "KESK", 4, 629.95, 4, 629.95),
    ("c208_21", "KESK", 4, 629.95, 4, 629.95),
    ("r101_21", "SCHI", 18, 1615.50, 17, 1624.89),
    ("r102_21", "HIER", 15, 1521.32, 15, 1454.53),
    ("r103_21", "SCHI", 13, 1244.15, 12, 1304.24),
    ("r104_21", "SCHI", 11, 1056.87, 11, 1051.41),
    ("r105_21", "SCHI", 14, 1347.80, 14, 1347.80),
    ("r106_21", "SCHI", 13, 1268.25, 13, 1263.13),
    ("r107_21", "SCHI", 12, 1110.95, 11, 1104.51),
    ("r108_21", "SCHI", 11, 1020.52, 10, 1030.44),
    ("r109_21", "SCHI", 12, 1186.99, 12, 1176.69),
    ("r110_21", "SCHI", 11, 1070.99, 11, 1067.11),
    ("r111_21", "HIER", 11, 1147.22, 11, 1076.15),
    ("r112_21", "SCHI", 11, 1001.79, 11, 1001.79),
    ("r201_21", "SCHI", 3, 1255.81, 3, 1255.81),
    ("r202_21", "HIER", 3, 1051.46, 3, 1052.52),
    ("r203_21", "KESK", 3, 895.54, 3, 895.54),
    ("r204_21", "SCHI", 2, 779.49, 2, 779.49),
    ("r205_21", "KESK", 3, 987.36, 3, 987.22),
    ("r206_21", "KESK", 3, 922.70, 3, 922.08),
    ("r207_21", "SCHI", 2, 843.20, 2, 845.19),
    ("r208_21", "KESK", 2, 736.12, 2, 736.46),
    ("r209_21", "SCHI", 3, 863.36, 3, 863.17),
    ("r210_21", "KESK", 3, 843.36, 3, 844.71),
    ("r211_21", "SCHI", 2, 827.29, 2, 825.25),
    ("rc101_21", "HIER", 15, 1725.73, 15, 1661.53),
    ("rc102_21", "KESK", 14, 1155.50, 14, 1510.16),
    ("rc103_21", "HIER", 12, 1388.72, 12, 1359.34),
    ("rc104_21", "SCHI", 11, 1175.06, 11, 1174.32),
    ("rc105_21", "SCHI", 14, 1450.82, 13, 1471.80),
    ("rc106_21", "SCHI", 13, 1385.96, 13, 1391.23),
    ("rc107_21", "SCHI", 12, 1250.30, 11, 1244.37),
    ("rc108_21", "SCHI", 11, 1154.14, 11, 1154.14),
    ("rc201_21", "SCHI", 4, 1445.17, 4, 1433.57),
    ("rc202_21", "SCHI", 3, 1408.08, 3, 1403.67),
    ("rc203_21", "SCHI", 3, 1060.32, 3, 1054.91),
    ("rc204_21", "SCHI", 3, 884.75, 3, 884.75),
    ("rc205_21", "SCHI", 3, 1259.69, 3, 1238.46),
    ("rc206_21", "SCHI", 3, 1189.11, 3, 1197.60),
    ("rc207_21", "HIER", 3, 985.67, 3, 978.30),
    ("rc208_21", "SCHI", 3, 833.12, 3, 833.12),
]

SOURCE_NAMES = {
    "SCHN": "Schneider et al. (2014)",
    "GOEK": "Goeke and Schneider",
    "HIE1": "Hiermann et al.",
    "HIE2": "Hiermann et al.",
    "HIER": "Hiermann et al.",
    "KESK": "Keskin and Catay (2016)",
    "SCHI": "Schiffer and Walther",
    "ERDE": "Erdelic and Caric (2022)",
}

CITATION = (
    "Erdelić, T.; Carić, T. Goods Delivery with Electric Vehicles: Electric "
    "Vehicle Routing Optimization with Time Windows and Partial or Full Recharge. "
    "Energies 2022, 15, 285. https://doi.org/10.3390/en15010285"
)

_NOTES = {
    "rc102_21": (
        "PR literature d=1155.50 is as printed in Erdelic 2022 Table 3; "
        "their ALNS is 1510.16 at K=14."
    ),
}


def _lex_pick(src_a: str, k_a: int, d_a: float, src_b: str, k_b: int, d_b: float) -> tuple[str, int, float]:
    if k_b < k_a:
        return src_b, k_b, d_b
    if k_a < k_b:
        return src_a, k_a, d_a
    if d_b + 1e-9 < d_a:
        return src_b, k_b, d_b
    return src_a, k_a, d_a


def bks_table() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, src, k, d, ek, ed in _PR:
        latest_src, lk, ld = _lex_pick(src, k, d, "ERDE", ek, ed)
        out[name] = {
            "instance": name,
            "literature_source": src,
            "literature_source_name": SOURCE_NAMES.get(src, src),
            "literature_k": k,
            "literature_d": d,
            "erdelic_k": ek,
            "erdelic_d": ed,
            "bks_source": latest_src,
            "bks_source_name": SOURCE_NAMES.get(latest_src, latest_src),
            "bks_k": lk,
            "bks_d": ld,
            "note": _NOTES.get(name, ""),
        }
    return out


def compare_row(instance: str, k: int, distance: float) -> dict[str, Any]:
    rec = bks_table()[instance]
    dk = int(k) - int(rec["bks_k"])
    gap = None
    if dk == 0 and rec["bks_d"]:
        gap = 100.0 * (float(distance) - float(rec["bks_d"])) / float(rec["bks_d"])
    if dk < 0:
        verdict = "fleet_win"
    elif dk > 0:
        verdict = "fleet_loss"
    elif gap is not None and gap < -1e-6:
        verdict = "distance_win"
    elif gap is not None and gap > 1e-6:
        verdict = "distance_loss"
    else:
        verdict = "tie"
    return {
        **rec,
        "our_k": int(k),
        "our_d": float(distance),
        "delta_k": dk,
        "distance_gap_pct": gap,
        "verdict": verdict,
    }
