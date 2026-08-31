"""Four matched metaheuristic baselines for the Hanif 1000-customer set."""
from src.hanif.benchmarks.alns import HYPERPARAMETERS as ALNS_HP
from src.hanif.benchmarks.alns import solve_alns
from src.hanif.benchmarks.greedy import HYPERPARAMETERS as GI_HP
from src.hanif.benchmarks.greedy import solve_greedy
from src.hanif.benchmarks.sa import HYPERPARAMETERS as SA_HP
from src.hanif.benchmarks.sa import solve_sa
from src.hanif.benchmarks.vns import HYPERPARAMETERS as VNS_HP
from src.hanif.benchmarks.vns import solve_vns

METHODS = {
    "GI": (solve_greedy, GI_HP),
    "ALNS": (solve_alns, ALNS_HP),
    "VNS": (solve_vns, VNS_HP),
    "SA": (solve_sa, SA_HP),
}

HYPERPARAMETERS = {
    "GI": GI_HP,
    "ALNS": ALNS_HP,
    "VNS": VNS_HP,
    "SA": SA_HP,
}

__all__ = [
    "METHODS",
    "HYPERPARAMETERS",
    "solve_greedy",
    "solve_alns",
    "solve_vns",
    "solve_sa",
]
