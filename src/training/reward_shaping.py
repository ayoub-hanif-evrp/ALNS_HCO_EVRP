"""Training-only reward shaping under hard original time windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
class StepContext:
    expired_before: int = 0
    expired_after: int = 0
    mask_counts: dict[str, float] = field(default_factory=dict)
    action_time_consumed: float = 0.0
    travel_time: float = 0.0
    waiting_time: float = 0.0
    service_time: float = 0.0
    charging_time: float = 0.0
    station_necessary: bool = True
    served_slack: float = 0.0
    mean_feasible_slack: float = 0.0
    min_feasible_slack: float = 0.0
    max_feasible_slack: float = 0.0
    urgency_rank: int = -1
    urgent_rank_k: int = 10
    customer_logit_bonus_applied: float = 0.0
    station_logit_penalty_applied: float = 0.0
    depot_logit_penalty_applied: float = 0.0
    steps_since_last_customer_service: int = 0


@dataclass
class TrainingRewardStats:
    served_bonus_total: float = 0.0
    urgency_bonus_total: float = 0.0
    new_customers_served: int = 0
    early_depot_returns: int = 0
    early_depot_penalty_total: float = 0.0
    step_penalty_total: float = 0.0
    expired_penalty_total: float = 0.0
    newly_expired_customers: int = 0
    cumulative_expired_customers: int = 0
    time_waste_penalty_total: float = 0.0
    unnecessary_station_actions: int = 0
    necessary_station_actions: int = 0
    unnecessary_station_penalty_total: float = 0.0
    urgent_rank_penalty_total: float = 0.0
    terminal_penalty_coef: float = 1.0
    terminal_unserved_penalty_raw: float = 0.0
    terminal_unserved_penalty_scaled: float = 0.0
    customer_actions: int = 0
    station_actions: int = 0
    depot_actions: int = 0
    shaped_return: float = 0.0
    raw_objective_return: float = 0.0
    served_slack_sum: float = 0.0
    served_slack_count: int = 0
    mask_reason_totals: dict[str, float] = field(default_factory=dict)
    no_progress_penalty_total: float = 0.0

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
    use_urgency_bonus: bool = True
    urgency_bonus_coef: float = 0.5
    use_newly_expired_customer_penalty: bool = True
    newly_expired_customer_penalty: float = 1.0
    use_time_waste_penalty: bool = True
    time_waste_penalty_coef: float = 0.05
    time_waste_scale: float = 1.0
    use_unnecessary_station_penalty: bool = True
    unnecessary_station_penalty: float = 1.0
    use_early_depot_penalty: bool = True
    early_depot_penalty_coef: float = 2.0
    use_step_penalty: bool = False
    step_penalty: float = 0.01
    use_urgent_customer_rank_penalty: bool = True
    urgent_rank_penalty_coef: float = 0.2
    urgent_rank_k: int = 10
    terminal_schedule: TerminalPenaltySchedule = field(default_factory=TerminalPenaltySchedule)
    use_no_progress_penalty: bool = True
    no_progress_threshold: int = 10
    no_progress_penalty_coef: float = 0.05
    use_customer_priority_logit_bias: bool = True
    customer_logit_bonus: float = 1.0
    unnecessary_station_logit_penalty: float = 1.5
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
            use_urgency_bonus=bool(getattr(rl_cfg, "use_urgency_bonus", True)),
            urgency_bonus_coef=float(getattr(rl_cfg, "urgency_bonus_coef", 0.5)),
            use_newly_expired_customer_penalty=bool(
                getattr(rl_cfg, "use_newly_expired_customer_penalty", True)
            ),
            newly_expired_customer_penalty=float(
                getattr(rl_cfg, "newly_expired_customer_penalty", 1.0)
            ),
            use_time_waste_penalty=bool(getattr(rl_cfg, "use_time_waste_penalty", True)),
            time_waste_penalty_coef=float(getattr(rl_cfg, "time_waste_penalty_coef", 0.05)),
            time_waste_scale=float(getattr(rl_cfg, "time_waste_scale", 1.0)),
            use_unnecessary_station_penalty=bool(
                getattr(rl_cfg, "use_unnecessary_station_penalty", True)
            ),
            unnecessary_station_penalty=float(getattr(rl_cfg, "unnecessary_station_penalty", 1.0)),
            use_early_depot_penalty=bool(getattr(rl_cfg, "use_early_depot_penalty", True)),
            early_depot_penalty_coef=float(getattr(rl_cfg, "early_depot_penalty_coef", 2.0)),
            use_step_penalty=bool(getattr(rl_cfg, "use_step_penalty", False)),
            step_penalty=float(getattr(rl_cfg, "step_penalty", 0.01)),
            use_urgent_customer_rank_penalty=bool(
                getattr(rl_cfg, "use_urgent_customer_rank_penalty", True)
            ),
            urgent_rank_penalty_coef=float(getattr(rl_cfg, "urgent_rank_penalty_coef", 0.2)),
            urgent_rank_k=int(getattr(rl_cfg, "urgent_rank_k", 10)),
            terminal_schedule=sched_raw,
            use_customer_priority_logit_bias=bool(
                getattr(rl_cfg, "use_customer_priority_logit_bias", True)
            ),
            customer_logit_bonus=float(getattr(rl_cfg, "customer_logit_bonus", 1.0)),
            unnecessary_station_logit_penalty=float(
                getattr(rl_cfg, "unnecessary_station_logit_penalty", 1.5)
            ),
            use_depot_logit_penalty=bool(getattr(rl_cfg, "use_depot_logit_penalty", True)),
            depot_logit_penalty=float(getattr(rl_cfg, "depot_logit_penalty", 2.0)),
            depot_penalty_until_served_ratio=float(
                getattr(rl_cfg, "depot_penalty_until_served_ratio", 0.95)
            ),
            use_no_progress_penalty=bool(getattr(rl_cfg, "use_no_progress_penalty", True)),
            no_progress_threshold=int(getattr(rl_cfg, "no_progress_threshold", 10)),
            no_progress_penalty_coef=float(getattr(rl_cfg, "no_progress_penalty_coef", 0.05)),
        )


class TrainingRewardCalculator:
    """Wraps objective-aligned step rewards with time-aware training bonuses/penalties."""

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

    def _urgency_bonus(self, ctx: StepContext) -> float:
        if not self.cfg.use_urgency_bonus or ctx.served_slack <= 0:
            return 0.0
        # Normalize tightness against the loosest feasible customer, not min (which equals served for ranked slack).
        ref_slack = max(ctx.max_feasible_slack, ctx.mean_feasible_slack, 1e-6)
        score = 1.0 - min(1.0, ctx.served_slack / ref_slack)
        if ctx.urgency_rank > 0 and ctx.urgency_rank <= 5:
            score = max(score, 1.0 - (ctx.urgency_rank - 1) / 5.0)
        return self.cfg.urgency_bonus_coef * self.served_bonus_value() * score

    def shape_step(
        self,
        step_info: StepInfo,
        *,
        before: CostSnapshot,
        after: CostSnapshot,
        node_type: str,
        customer_served: bool,
        ctx: StepContext | None = None,
    ) -> StepInfo:
        ctx = ctx or StepContext()
        scale = self.cfg.reward_scale
        bonus = 0.0
        urgency = 0.0
        delta_served = after.num_served_customers - before.num_served_customers
        if self.cfg.use_served_customer_bonus and delta_served > 0:
            bonus = self.served_bonus_value() * delta_served
            urgency = self._urgency_bonus(ctx) * delta_served
            self.stats.new_customers_served += int(delta_served)
            self.stats.served_bonus_total += bonus
            self.stats.urgency_bonus_total += urgency
            if ctx.served_slack > 0:
                self.stats.served_slack_sum += ctx.served_slack
                self.stats.served_slack_count += 1

        expired_pen = 0.0
        newly = max(0, ctx.expired_after - ctx.expired_before)
        if self.cfg.use_newly_expired_customer_penalty and newly > 0:
            expired_pen = (
                self.cfg.newly_expired_customer_penalty
                * self.weights.penalty_unserved
                * newly
                / scale
            )
            self.stats.newly_expired_customers += newly
            self.stats.cumulative_expired_customers += newly
            self.stats.expired_penalty_total += expired_pen

        time_waste_pen = 0.0
        if self.cfg.use_time_waste_penalty and ctx.action_time_consumed > 0:
            time_waste_pen = (
                self.cfg.time_waste_penalty_coef
                * ctx.action_time_consumed
                / max(self.cfg.time_waste_scale, 1e-6)
                / scale
            )
            self.stats.time_waste_penalty_total += time_waste_pen

        station_pen = 0.0
        if node_type == "station" and self.cfg.use_unnecessary_station_penalty:
            if ctx.station_necessary:
                self.stats.necessary_station_actions += 1
            else:
                self.stats.unnecessary_station_actions += 1
                station_pen = self.cfg.unnecessary_station_penalty / scale
                self.stats.unnecessary_station_penalty_total += station_pen

        early_pen = 0.0
        if self.cfg.use_early_depot_penalty and node_type in ("route_close", "depot_finish"):
            unserved_before = self.num_customers - before.num_served_customers
            if unserved_before > 0:
                unserved_ratio = unserved_before / self.num_customers
                early_pen = (
                    self.cfg.early_depot_penalty_coef
                    * self.weights.penalty_unserved
                    * unserved_ratio
                    / scale
                )
                self.stats.early_depot_returns += 1
                self.stats.early_depot_penalty_total += early_pen

        rank_pen = 0.0
        if (
            self.cfg.use_urgent_customer_rank_penalty
            and customer_served
            and ctx.urgency_rank > self.cfg.urgent_rank_k
            and ctx.min_feasible_slack >= 0
        ):
            rank_pen = self.cfg.urgent_rank_penalty_coef / scale
            self.stats.urgent_rank_penalty_total += rank_pen

        step_pen = 0.0
        if self.cfg.use_step_penalty and not customer_served:
            step_pen = self.cfg.step_penalty / scale
            self.stats.step_penalty_total += step_pen

        no_prog_pen = 0.0
        if (
            self.cfg.use_no_progress_penalty
            and ctx.steps_since_last_customer_service > self.cfg.no_progress_threshold
        ):
            excess = ctx.steps_since_last_customer_service - self.cfg.no_progress_threshold
            no_prog_pen = self.cfg.no_progress_penalty_coef * excess / scale
            self.stats.no_progress_penalty_total += no_prog_pen

        for k, v in ctx.mask_counts.items():
            self.stats.mask_reason_totals[k] = self.stats.mask_reason_totals.get(k, 0.0) + float(v)

        shaped = (
            step_info.reward + bonus + urgency - expired_pen - time_waste_pen
            - station_pen - early_pen - rank_pen - step_pen - no_prog_pen
        )
        step_info.reward = shaped
        extra = dict(step_info.extra or {})
        extra.update(
            {
                "served_bonus": bonus,
                "urgency_bonus": urgency,
                "expired_customer_penalty": expired_pen,
                "newly_expired_customers": float(newly),
                "time_waste_penalty": time_waste_pen,
                "action_time_consumed": ctx.action_time_consumed,
                "unnecessary_station_penalty": station_pen,
                "early_depot_penalty": early_pen,
                "urgent_rank_penalty": rank_pen,
                "no_progress_penalty": no_prog_pen,
                "steps_since_last_customer_service": float(ctx.steps_since_last_customer_service),
                "raw_step_reward": step_info.raw_reward,
                "served_customer_slack": ctx.served_slack,
                "mean_feasible_customer_slack": ctx.mean_feasible_slack,
                "min_feasible_customer_slack": ctx.min_feasible_slack,
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
        return step_info

    def terminal_adjustment(self, before: CostSnapshot, summary: SolutionSummary) -> StepInfo:
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

    def training_logit_adjustments(
        self,
        env,
        ev,
        feasible_mask,
        *,
        unserved_ratio: float,
    ) -> tuple[float, float, float]:
        """Return (customer_bonus, station_penalty, depot_penalty) for logit bias."""
        from src.env.tw_diagnostics import selected_station_is_necessary

        cust_bonus = 0.0
        st_pen = 0.0
        dep_pen = self.depot_logit_penalty(env) if self.cfg.use_depot_logit_penalty else 0.0
        inst = env.instance
        any_customer = any(
            feasible_mask[j] for j in inst.customer_indices if not env.served_mask[j]
        )
        if self.cfg.use_customer_priority_logit_bias and any_customer:
            cust_bonus = self.cfg.customer_logit_bonus
        if self.cfg.use_customer_priority_logit_bias:
            for s in inst.charging_station_indices:
                if feasible_mask[s]:
                    necessary, _ = selected_station_is_necessary(env, ev, s)
                    if not necessary:
                        st_pen = max(st_pen, self.cfg.unnecessary_station_logit_penalty)
        self.stats.mask_reason_totals["customer_logit_bonus_applied"] = (
            self.stats.mask_reason_totals.get("customer_logit_bonus_applied", 0.0) + cust_bonus
        )
        self.stats.mask_reason_totals["station_logit_penalty_applied"] = (
            self.stats.mask_reason_totals.get("station_logit_penalty_applied", 0.0) + st_pen
        )
        self.stats.mask_reason_totals["depot_logit_penalty_applied"] = (
            self.stats.mask_reason_totals.get("depot_logit_penalty_applied", 0.0) + dep_pen
        )
        return cust_bonus, st_pen, dep_pen

    def depot_logit_penalty(self, env) -> float:
        if not self.cfg.use_depot_logit_penalty:
            return 0.0
        served_ratio = env.num_served_customers / self.num_customers
        if served_ratio >= self.cfg.depot_penalty_until_served_ratio:
            return 0.0
        return float(self.cfg.depot_logit_penalty * (1.0 - served_ratio))

    def finalize(self, final_objective: float) -> dict[str, float]:
        self.stats.raw_objective_return = -float(final_objective) / self.cfg.reward_scale
        out: dict[str, float] = {
            "final_original_objective": float(final_objective),
            "served_bonus_total": self.stats.served_bonus_total,
            "urgency_bonus_total": self.stats.urgency_bonus_total,
            "new_customers_served": float(self.stats.new_customers_served),
            "early_depot_returns": float(self.stats.early_depot_returns),
            "early_depot_penalty_total": self.stats.early_depot_penalty_total,
            "expired_customer_penalty_total": self.stats.expired_penalty_total,
            "newly_expired_customers": float(self.stats.newly_expired_customers),
            "cumulative_expired_customers": float(self.stats.cumulative_expired_customers),
            "time_waste_penalty_total": self.stats.time_waste_penalty_total,
            "unnecessary_station_actions": float(self.stats.unnecessary_station_actions),
            "necessary_station_actions": float(self.stats.necessary_station_actions),
            "unnecessary_station_penalty_total": self.stats.unnecessary_station_penalty_total,
            "urgent_rank_penalty_total": self.stats.urgent_rank_penalty_total,
            "step_penalty_total": self.stats.step_penalty_total,
            "terminal_penalty_coef": self.stats.terminal_penalty_coef,
            "terminal_unserved_penalty_raw": self.stats.terminal_unserved_penalty_raw,
            "terminal_unserved_penalty_scaled": self.stats.terminal_unserved_penalty_scaled,
            "customer_actions": float(self.stats.customer_actions),
            "station_actions": float(self.stats.station_actions),
            "depot_actions": float(self.stats.depot_actions),
            "shaped_training_return": self.stats.shaped_return,
            "raw_objective_return": self.stats.raw_objective_return,
            "served_customer_slack": (
                self.stats.served_slack_sum / max(self.stats.served_slack_count, 1)
            ),
            **self.stats.action_fractions(
                self.stats.customer_actions + self.stats.station_actions + self.stats.depot_actions
            ),
        }
        for k, v in self.stats.mask_reason_totals.items():
            out[k] = v
        decisions = max(
            self.stats.customer_actions + self.stats.station_actions + self.stats.depot_actions, 1
        )
        for key in (
            "feasible_customers_count",
            "masked_by_time_window_count",
            "feasible_customer_ratio",
            "time_window_masked_customer_ratio",
        ):
            if key in out:
                out[f"mean_{key}"] = float(out[key]) / decisions
        return out
