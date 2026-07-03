"""Training-only reward shaping (evaluation uses raw EVRP objective)."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.env.cost import CostWeights
from src.env.reward import CostSnapshot, StepInfo, StepRewardCalculator, SolutionSummary


@dataclass
class TerminalPenaltySchedule:
    enabled: bool = False
    start_coef: float = 0.25
    end_coef: float = 1.0
    warmup_cycles: int = 20

    def coef_for_cycle(self, cycle_id: int) -> float:
        if not self.enabled:
            return 1.0
        if self.warmup_cycles <= 0:
            return float(self.end_coef)
        t = min(1.0, max(0.0, (max(1, cycle_id) - 1) / float(self.warmup_cycles)))
        return float(self.start_coef + t * (self.end_coef - self.start_coef))


@dataclass
class TrainingRewardStats:
    served_bonus_total: float = 0.0
    new_customers_served: int = 0
    early_depot_returns: int = 0
    early_depot_penalty_total: float = 0.0
    step_penalty_total: float = 0.0
    terminal_penalty_coef: float = 1.0
    terminal_unserved_penalty_raw: float = 0.0
    terminal_unserved_penalty_scaled: float = 0.0
    customer_actions: int = 0
    station_actions: int = 0
    depot_actions: int = 0
    depot_actions_before_all_served: int = 0
    shaped_return: float = 0.0
    raw_objective_return: float = 0.0

    def action_fractions(self, num_decisions: int) -> dict[str, float]:
        d = max(num_decisions, 1)
        return {
            "fraction_customer_actions": self.customer_actions / d,
            "fraction_station_actions": self.station_actions / d,
            "fraction_depot_actions": self.depot_actions / d,
        }


@dataclass
class TrainingRewardConfig:
    reward_scale: float = 1000.0
    use_served_customer_bonus: bool = True
    served_customer_bonus: float | None = None
    served_bonus_mode: str = "penalty_equivalent"
    served_bonus_coef: float = 0.5
    use_early_depot_penalty: bool = True
    early_depot_penalty_coef: float = 1.0
    use_step_penalty: bool = True
    step_penalty: float = 0.01
    terminal_schedule: TerminalPenaltySchedule = field(default_factory=TerminalPenaltySchedule)
    use_depot_logit_penalty: bool = True
    depot_logit_penalty: float = 2.0
    depot_penalty_until_served_ratio: float = 0.95

    @classmethod
    def from_rl_config(cls, rl_cfg) -> "TrainingRewardConfig":
        sched_raw = getattr(rl_cfg, "terminal_penalty_schedule", None) or TerminalPenaltySchedule()
        return cls(
            reward_scale=float(rl_cfg.reward_scale),
            use_served_customer_bonus=bool(getattr(rl_cfg, "use_served_customer_bonus", True)),
            served_customer_bonus=getattr(rl_cfg, "served_customer_bonus", None),
            served_bonus_mode=str(getattr(rl_cfg, "served_bonus_mode", "penalty_equivalent")),
            served_bonus_coef=float(getattr(rl_cfg, "served_bonus_coef", 0.5)),
            use_early_depot_penalty=bool(getattr(rl_cfg, "use_early_depot_penalty", True)),
            early_depot_penalty_coef=float(getattr(rl_cfg, "early_depot_penalty_coef", 1.0)),
            use_step_penalty=bool(getattr(rl_cfg, "use_step_penalty", True)),
            step_penalty=float(getattr(rl_cfg, "step_penalty", 0.01)),
            terminal_schedule=sched_raw,
            use_depot_logit_penalty=bool(getattr(rl_cfg, "use_depot_logit_penalty", True)),
            depot_logit_penalty=float(getattr(rl_cfg, "depot_logit_penalty", 2.0)),
            depot_penalty_until_served_ratio=float(
                getattr(rl_cfg, "depot_penalty_until_served_ratio", 0.95)
            ),
        )


class TrainingRewardCalculator:
    """Wraps objective-aligned step rewards with dense training bonuses/penalties."""

    def __init__(
        self,
        weights: CostWeights,
        cfg: TrainingRewardConfig,
        *,
        num_customers: int,
        cycle_id: int = 1,
    ):
        self.weights = weights
        self.cfg = cfg
        self.num_customers = max(int(num_customers), 1)
        self.cycle_id = max(1, int(cycle_id))
        self.base = StepRewardCalculator(weights, reward_scale=cfg.reward_scale)
        self.stats = TrainingRewardStats(
            terminal_penalty_coef=cfg.terminal_schedule.coef_for_cycle(self.cycle_id),
        )

    def served_bonus_value(self) -> float:
        if self.cfg.served_customer_bonus is not None:
            return float(self.cfg.served_customer_bonus) / self.cfg.reward_scale
        if self.cfg.served_bonus_mode == "penalty_equivalent":
            return (
                self.cfg.served_bonus_coef
                * self.weights.penalty_unserved
                / self.cfg.reward_scale
            )
        return float(self.cfg.served_bonus_coef) / self.cfg.reward_scale

    def shape_step(
        self,
        step_info: StepInfo,
        *,
        before: CostSnapshot,
        after: CostSnapshot,
        node_type: str,
        customer_served: bool,
    ) -> StepInfo:
        """Apply dense training bonuses/penalties on top of objective-aligned step reward."""
        bonus = 0.0
        delta_served = after.num_served_customers - before.num_served_customers
        if self.cfg.use_served_customer_bonus and delta_served > 0:
            bonus = self.served_bonus_value() * delta_served
            self.stats.new_customers_served += int(delta_served)
            self.stats.served_bonus_total += bonus

        early_pen = 0.0
        if self.cfg.use_early_depot_penalty and node_type == "route_close":
            unserved_before = self.num_customers - before.num_served_customers
            if unserved_before > 0:
                unserved_ratio = unserved_before / self.num_customers
                early_pen = (
                    self.cfg.early_depot_penalty_coef
                    * self.weights.penalty_unserved
                    * unserved_ratio
                    / self.cfg.reward_scale
                )
                self.stats.early_depot_returns += 1
                self.stats.early_depot_penalty_total += early_pen

        step_pen = 0.0
        if self.cfg.use_step_penalty and not customer_served:
            step_pen = self.cfg.step_penalty / self.cfg.reward_scale
            self.stats.step_penalty_total += step_pen

        shaped = step_info.reward + bonus - early_pen - step_pen
        step_info.reward = shaped
        extra = dict(step_info.extra or {})
        extra.update(
            {
                "served_bonus": bonus,
                "early_depot_penalty": early_pen,
                "step_penalty": step_pen,
                "raw_step_reward": step_info.raw_reward,
            }
        )
        step_info.extra = extra
        self.stats.shaped_return += shaped

        if node_type == "customer":
            self.stats.customer_actions += 1
        elif node_type == "station":
            self.stats.station_actions += 1
        elif node_type in ("depot", "depot_recharge"):
            self.stats.depot_actions += 1
            if before.num_served_customers < self.num_customers:
                self.stats.depot_actions_before_all_served += 1
        return step_info

    def terminal_adjustment(self, before: CostSnapshot, summary: SolutionSummary) -> StepInfo:
        """Terminal reward with scheduled unserved penalty (training only)."""
        w = self.weights
        coef = self.cfg.terminal_schedule.coef_for_cycle(self.cycle_id)
        self.stats.terminal_penalty_coef = coef

        unserved_raw = w.penalty_unserved * summary.unserved_customers
        unserved_scaled = coef * unserved_raw
        self.stats.terminal_unserved_penalty_raw = float(unserved_raw)
        self.stats.terminal_unserved_penalty_scaled = float(unserved_scaled)

        other = (
            w.penalty_reserved * summary.unresolved_reserved_customers
            + w.penalty_infeasible * summary.infeasible_flag
            + w.lambda_vehicle * max(0, summary.used_evs - before.used_evs)
        )
        terminal_j = unserved_scaled + other
        reward = -terminal_j / self.cfg.reward_scale

        info = StepInfo(
            ev_id=-1,
            selected_node=-1,
            node_type="terminal",
            previous_node=-1,
            immediate_penalty=terminal_j,
            delta_objective=terminal_j,
            reward=reward,
            raw_reward=reward,
            terminal=True,
            extra={
                "terminal_penalty_cost": terminal_j,
                "terminal_unserved_penalty_raw": unserved_raw,
                "terminal_unserved_penalty_scaled": unserved_scaled,
                "terminal_penalty_coef": coef,
            },
        )
        self.stats.shaped_return += reward
        return info

    def depot_logit_penalty(self, env) -> float:
        if not self.cfg.use_depot_logit_penalty:
            return 0.0
        served_ratio = env.num_served_customers / self.num_customers
        if served_ratio >= self.cfg.depot_penalty_until_served_ratio:
            return 0.0
        unserved_ratio = 1.0 - served_ratio
        return float(self.cfg.depot_logit_penalty * unserved_ratio)

    def finalize(self, final_objective: float) -> dict[str, float]:
        self.stats.raw_objective_return = -float(final_objective) / self.cfg.reward_scale
        out = {
            "served_bonus_total": self.stats.served_bonus_total,
            "served_bonus_per_episode": self.stats.served_bonus_total,
            "customers_served_this_episode": float(self.stats.new_customers_served),
            "new_customer_served_count": float(self.stats.new_customers_served),
            "early_depot_returns": float(self.stats.early_depot_returns),
            "early_depot_penalty_total": self.stats.early_depot_penalty_total,
            "step_penalty_total": self.stats.step_penalty_total,
            "terminal_penalty_coef": self.stats.terminal_penalty_coef,
            "terminal_unserved_penalty_raw": self.stats.terminal_unserved_penalty_raw,
            "terminal_unserved_penalty_scaled": self.stats.terminal_unserved_penalty_scaled,
            "customer_actions": float(self.stats.customer_actions),
            "station_actions": float(self.stats.station_actions),
            "depot_actions": float(self.stats.depot_actions),
            "depot_actions_before_all_served": float(self.stats.depot_actions_before_all_served),
            "shaped_training_return": self.stats.shaped_return,
            "raw_objective_return": self.stats.raw_objective_return,
            **self.stats.action_fractions(
                self.stats.customer_actions + self.stats.station_actions + self.stats.depot_actions
            ),
        }
        return out
