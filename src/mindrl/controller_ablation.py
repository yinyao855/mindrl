"""Lightweight controller ablations for branch-native reward interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mindrl.core import AlgorithmConfig, TrainReport
from mindrl.interface_controller import (
    AdapterDecision,
    BarrierProfile,
    MindRLController,
    PolicySpec,
)

Strategy = Literal["uniform", "score_routing", "barrier_gated"]


@dataclass(frozen=True)
class BranchScenario:
    spec: PolicySpec
    profile: BarrierProfile
    reward: float


@dataclass(frozen=True)
class AblationResult:
    strategy: Strategy
    effective_reward: float
    risk_cost: float
    update_budget: float
    decisions: tuple[AdapterDecision, ...]


def run_controller_ablation(
    scenarios: list[BranchScenario],
    strategy: Strategy,
    controller: MindRLController | None = None,
) -> AblationResult:
    """Evaluate one reward-routing strategy on branch scenarios."""

    if not scenarios:
        raise ValueError("scenarios must not be empty")
    controller = controller or MindRLController()
    decisions: list[AdapterDecision] = []
    effective_reward = 0.0
    risk_cost = 0.0
    update_budget = 0.0

    for scenario in scenarios:
        decision = _decision_for_strategy(scenario, strategy, controller)
        decisions.append(decision)
        branch_risk = _profile_risk(scenario.profile)
        effective_reward += scenario.reward * decision.branch_weight
        risk_cost += (
            branch_risk
            * decision.branch_weight
            * decision.clip_range
            / max(1.0, decision.anchor_strength)
        )
        update_budget += decision.granularity * decision.branch_weight

    count = len(scenarios)
    return AblationResult(
        strategy=strategy,
        effective_reward=effective_reward / count,
        risk_cost=risk_cost / count,
        update_budget=update_budget / count,
        decisions=tuple(decisions),
    )


def summarize_controller_ablation(
    run_name: str,
    scenarios: list[BranchScenario],
) -> TrainReport:
    """Run all controller strategies and return a compact report."""

    strategies: tuple[Strategy, ...] = ("uniform", "score_routing", "barrier_gated")
    results = {
        strategy: run_controller_ablation(scenarios, strategy)
        for strategy in strategies
    }
    metrics: dict[str, float] = {}
    for strategy, result in results.items():
        metrics[f"{strategy}_effective_reward"] = result.effective_reward
        metrics[f"{strategy}_risk_cost"] = result.risk_cost
        metrics[f"{strategy}_update_budget"] = result.update_budget
    best_strategy = min(
        strategies,
        key=lambda strategy: (
            results[strategy].risk_cost,
            -results[strategy].effective_reward,
        ),
    )
    return TrainReport(
        run_name=run_name,
        algorithm=AlgorithmConfig(name="controller_ablation", branch="mixed"),
        metrics=metrics,
        artifacts={"best_strategy": best_strategy},
    )


def _decision_for_strategy(
    scenario: BranchScenario,
    strategy: Strategy,
    controller: MindRLController,
) -> AdapterDecision:
    if strategy == "barrier_gated":
        return controller.decide(scenario.spec, scenario.profile)
    if strategy == "score_routing":
        return AdapterDecision(
            adapter="score_routing_only",
            granularity=scenario.spec.default_granularity,
            anchor_strength=0.0,
            structure_cost_weight=0.0,
            branch_weight=1.0 if scenario.spec.score_availability == "exact" else 0.5,
            clip_range=0.1,
        )
    if strategy == "uniform":
        return AdapterDecision(
            adapter="uniform_reward",
            granularity=scenario.spec.default_granularity,
            anchor_strength=0.0,
            structure_cost_weight=0.0,
            branch_weight=1.0,
            clip_range=0.2,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def _profile_risk(profile: BarrierProfile) -> float:
    return sum(
        value
        for value in [
            profile.nctc,
            profile.density_cost,
            profile.surrogate_variance,
            profile.drift,
            profile.smoothness,
        ]
        if value is not None
    )
