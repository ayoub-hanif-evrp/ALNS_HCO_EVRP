"""Per-EV state used by the asynchronous event-driven environment."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EVState:
    """State of a single electric vehicle (agent).

    Times are continuous; ``current_time`` is the EV's own clock. The EV that
    becomes available earliest acts next in the event-driven decoder.
    """

    ev_id: int
    current_node: int
    current_time: float
    remaining_battery: float
    remaining_capacity: float
    route_distance: float = 0.0
    total_waiting_time: float = 0.0
    total_charging_time: float = 0.0
    total_charging_price: float = 0.0
    route_nodes: list[int] = field(default_factory=list)
    arrival_times: list[float] = field(default_factory=list)
    departure_times: list[float] = field(default_factory=list)
    battery_trace: list[float] = field(default_factory=list)
    capacity_trace: list[float] = field(default_factory=list)
    # charging_actions: list of dicts {node, portion, charged_energy, charging_time}
    charging_actions: list[dict] = field(default_factory=list)
    active: bool = False
    busy: bool = False
    finished: bool = False
    current_reserved_customer: int | None = None

    @classmethod
    def initial(cls, ev_id: int, depot_index: int, battery_capacity: float, load_capacity: float) -> "EVState":
        return cls(
            ev_id=ev_id,
            current_node=depot_index,
            current_time=0.0,
            remaining_battery=battery_capacity,
            remaining_capacity=load_capacity,
            route_distance=0.0,
            route_nodes=[depot_index],
            arrival_times=[0.0],
            departure_times=[0.0],
            battery_trace=[battery_capacity],
            capacity_trace=[load_capacity],
            active=False,
            busy=False,
            finished=False,
            current_reserved_customer=None,
        )

    @property
    def num_customers_served(self) -> int:
        """Number of distinct route stops that are not the depot/station start."""
        return len(self.route_nodes) - 1

    def used(self) -> bool:
        """An EV is 'used' if it ever left the depot."""
        return len(self.route_nodes) > 1
