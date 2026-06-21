"""Flow/diffusion branch probes and adapters for MINDRL pilots."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from mindrl_repo.interface_controller import (
    AdapterDecision,
    BarrierProfile,
    MindRLController,
    PolicySpec,
)


@dataclass(frozen=True)
class FlowDiffusionTrace:
    """One flow/diffusion branch evaluation trace.

    The fields are deliberately model-agnostic so they can be populated by a
    toy simulator, a diffusers pipeline, UniRL rollout metadata, or a VLA action
    head.
    """

    reward: float
    surrogate_scores: tuple[float, ...]
    anchor_distances: tuple[float, ...]
    smoothness_costs: tuple[float, ...] = ()
    density_eval_seconds: float | None = None
    collision_or_feasibility_failures: int = 0


@dataclass(frozen=True)
class FlowDiffusionSummary:
    surrogate_variance: float
    drift: float
    smoothness: float
    density_cost: float
    reward: float
    failure_rate: float


@dataclass(frozen=True)
class FlowAdapterEvaluation:
    decision: AdapterDecision
    summary: FlowDiffusionSummary
    risk_adjusted_reward: float


def summarize_flow_trace(trace: FlowDiffusionTrace) -> FlowDiffusionSummary:
    """Convert raw flow/diffusion trace metrics into branch barriers."""

    if not trace.surrogate_scores:
        raise ValueError("surrogate_scores must not be empty")
    if not trace.anchor_distances:
        raise ValueError("anchor_distances must not be empty")

    smoothness = mean(trace.smoothness_costs) if trace.smoothness_costs else 0.0
    failure_rate = min(1.0, max(0.0, trace.collision_or_feasibility_failures / 10.0))
    return FlowDiffusionSummary(
        surrogate_variance=pstdev(trace.surrogate_scores),
        drift=mean(trace.anchor_distances),
        smoothness=smoothness,
        density_cost=trace.density_eval_seconds or 0.0,
        reward=trace.reward,
        failure_rate=failure_rate,
    )


def flow_barrier_profile(summary: FlowDiffusionSummary) -> BarrierProfile:
    """Map flow/diffusion summary statistics to the generic MINDRL profile."""

    return BarrierProfile(
        density_cost=summary.density_cost,
        surrogate_variance=summary.surrogate_variance,
        drift=summary.drift + summary.failure_rate,
        smoothness=summary.smoothness,
    )


def evaluate_flow_adapter(
    trace: FlowDiffusionTrace,
    spec: PolicySpec | None = None,
    controller: MindRLController | None = None,
) -> FlowAdapterEvaluation:
    """Evaluate the controller decision for one flow/diffusion branch."""

    summary = summarize_flow_trace(trace)
    profile = flow_barrier_profile(summary)
    spec = spec or PolicySpec(
        name="flow_action",
        modality="continuous",
        structure="flow",
        score_availability="surrogate",
        default_granularity=8,
    )
    decision = (controller or MindRLController()).decide(spec, profile)
    risk_penalty = (
        profile.surrogate_variance or 0.0
    ) * decision.clip_range + (profile.drift or 0.0) / max(1.0, decision.anchor_strength)
    risk_penalty += (profile.smoothness or 0.0) * decision.structure_cost_weight
    return FlowAdapterEvaluation(
        decision=decision,
        summary=summary,
        risk_adjusted_reward=summary.reward * decision.branch_weight - risk_penalty,
    )


def compare_flow_strategies(trace: FlowDiffusionTrace) -> dict[str, FlowAdapterEvaluation]:
    """Compare score-routing and full barrier-gated decisions for one trace."""

    controller = MindRLController()
    spec = PolicySpec(
        name="flow_action",
        modality="continuous",
        structure="flow",
        score_availability="surrogate",
        default_granularity=8,
    )
    gated = evaluate_flow_adapter(trace, spec=spec, controller=controller)
    score_routing_decision = AdapterDecision(
        adapter="score_routing_only",
        granularity=spec.default_granularity,
        anchor_strength=0.0,
        structure_cost_weight=0.0,
        branch_weight=0.5,
        clip_range=0.1,
    )
    summary = summarize_flow_trace(trace)
    profile = flow_barrier_profile(summary)
    score_routing_penalty = (
        (profile.surrogate_variance or 0.0) * score_routing_decision.clip_range
        + (profile.drift or 0.0)
    )
    return {
        "score_routing": FlowAdapterEvaluation(
            decision=score_routing_decision,
            summary=summary,
            risk_adjusted_reward=summary.reward * 0.5 - score_routing_penalty,
        ),
        "barrier_gated": gated,
    }
