"""Hybrid AR+flow/VLA interface utilities for MINDRL pilots."""

from __future__ import annotations

from dataclasses import dataclass

from mindrl.flow_diffusion_interface import (
    FlowDiffusionTrace,
    evaluate_flow_adapter,
    summarize_flow_trace,
)
from mindrl.interface_controller import (
    AdapterDecision,
    BarrierProfile,
    MindRLController,
    PolicySpec,
)


@dataclass(frozen=True)
class ARBranchTrace:
    token_reward: float
    exact_logprob_available: bool = True
    nctc: float | None = None


@dataclass(frozen=True)
class VLABranchTrace:
    task_success_proxy: float
    ar: ARBranchTrace
    flow: FlowDiffusionTrace
    semantic_staleness: float = 0.0
    world_uncertainty: float = 0.0


@dataclass(frozen=True)
class VLAControllerResult:
    ar_decision: AdapterDecision
    flow_decision: AdapterDecision
    task_success_proxy: float
    risk_adjusted_score: float
    semantic_refresh_budget: float
    action_chunk_granularity: int


def evaluate_vla_controller(
    trace: VLABranchTrace,
    controller: MindRLController | None = None,
) -> VLAControllerResult:
    """Evaluate a two-branch AR+flow controller decision."""

    controller = controller or MindRLController()
    ar_spec = PolicySpec(
        name="language_plan",
        modality="text",
        structure="ar" if trace.ar.exact_logprob_available else "parallel_discrete",
        score_availability="exact" if trace.ar.exact_logprob_available else "marginal",
        default_granularity=1,
    )
    ar_profile = BarrierProfile(nctc=trace.ar.nctc)
    ar_decision = controller.decide(ar_spec, ar_profile)

    flow_eval = evaluate_flow_adapter(trace.flow, controller=controller)
    semantic_penalty = trace.semantic_staleness + trace.world_uncertainty
    semantic_refresh_budget = min(1.0, semantic_penalty)
    risk_adjusted_score = (
        trace.task_success_proxy
        + trace.ar.token_reward * ar_decision.branch_weight
        + flow_eval.risk_adjusted_reward
        - semantic_penalty
    )
    return VLAControllerResult(
        ar_decision=ar_decision,
        flow_decision=flow_eval.decision,
        task_success_proxy=trace.task_success_proxy,
        risk_adjusted_score=risk_adjusted_score,
        semantic_refresh_budget=semantic_refresh_budget,
        action_chunk_granularity=flow_eval.decision.granularity,
    )


def compare_vla_strategies(trace: VLABranchTrace) -> dict[str, VLAControllerResult]:
    """Compare a naive score-routing setting against full barrier gating."""

    gated = evaluate_vla_controller(trace)
    ar_decision = AdapterDecision(
        adapter="exact_ratio" if trace.ar.exact_logprob_available else "score_routing_only",
        granularity=1,
        anchor_strength=0.0,
        structure_cost_weight=0.0,
        branch_weight=1.0,
        clip_range=0.2,
    )
    flow_decision = AdapterDecision(
        adapter="score_routing_only",
        granularity=8,
        anchor_strength=0.0,
        structure_cost_weight=0.0,
        branch_weight=0.5,
        clip_range=0.1,
    )
    flow_summary = summarize_flow_trace(trace.flow)
    uncontrolled_risk = (
        flow_summary.surrogate_variance * flow_decision.clip_range
        + flow_summary.drift
        + flow_summary.failure_rate
        + trace.semantic_staleness
        + trace.world_uncertainty
    )
    score_routing = VLAControllerResult(
        ar_decision=ar_decision,
        flow_decision=flow_decision,
        task_success_proxy=trace.task_success_proxy,
        risk_adjusted_score=(
            trace.task_success_proxy
            + trace.ar.token_reward * ar_decision.branch_weight
            + trace.flow.reward * flow_decision.branch_weight
            - uncontrolled_risk
        ),
        semantic_refresh_budget=0.0,
        action_chunk_granularity=flow_decision.granularity,
    )
    return {
        "score_routing_only": score_routing,
        "barrier_gated": gated,
    }
