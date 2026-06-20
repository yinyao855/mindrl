"""Pluggable MINDRL interface controller primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    """Static branch metadata exposed by an RL framework or rollout trace."""

    name: str
    modality: str
    structure: str
    score_availability: str
    default_granularity: int = 1


@dataclass(frozen=True)
class BarrierProfile:
    """Live barrier measurements for one policy branch."""

    nctc: float | None = None
    density_cost: float | None = None
    surrogate_variance: float | None = None
    drift: float | None = None
    smoothness: float | None = None


@dataclass(frozen=True)
class AdapterDecision:
    """Controller output consumed by an optimizer or training stack."""

    adapter: str
    granularity: int
    anchor_strength: float
    structure_cost_weight: float
    branch_weight: float
    clip_range: float


class MindRLController:
    """Rule-based reference controller for branch-native reward interfaces."""

    def __init__(
        self,
        high_nctc: float = 0.30,
        high_surrogate_variance: float = 0.20,
        high_drift: float = 0.40,
    ) -> None:
        self.high_nctc = high_nctc
        self.high_surrogate_variance = high_surrogate_variance
        self.high_drift = high_drift

    def decide(self, spec: PolicySpec, profile: BarrierProfile) -> AdapterDecision:
        if spec.structure == "ar" and spec.score_availability == "exact":
            return AdapterDecision(
                adapter="exact_ratio",
                granularity=spec.default_granularity,
                anchor_strength=0.0,
                structure_cost_weight=0.0,
                branch_weight=1.0,
                clip_range=0.2,
            )

        if spec.structure == "parallel_discrete":
            nctc = profile.nctc or 0.0
            if nctc >= self.high_nctc:
                granularity = max(1, spec.default_granularity // 4)
                structure_weight = 1.0
            else:
                granularity = spec.default_granularity
                structure_weight = 0.1
            return AdapterDecision(
                adapter="dependence_aware_block",
                granularity=granularity,
                anchor_strength=0.0,
                structure_cost_weight=structure_weight,
                branch_weight=1.0,
                clip_range=0.2,
            )

        if spec.structure in {"flow", "diffusion"}:
            surrogate_variance = profile.surrogate_variance or 0.0
            drift = profile.drift or 0.0
            anchor = 1.0
            branch_weight = 1.0
            clip_range = 0.2
            if drift >= self.high_drift:
                anchor = 2.0
            if surrogate_variance >= self.high_surrogate_variance:
                branch_weight = 0.5
                clip_range = 0.1
            return AdapterDecision(
                adapter="anchored_flow_surrogate",
                granularity=spec.default_granularity,
                anchor_strength=anchor,
                structure_cost_weight=1.0 if (profile.smoothness or 0.0) > 0 else 0.0,
                branch_weight=branch_weight,
                clip_range=clip_range,
            )

        return AdapterDecision(
            adapter="score_routing_only",
            granularity=spec.default_granularity,
            anchor_strength=0.0,
            structure_cost_weight=0.0,
            branch_weight=0.5,
            clip_range=0.1,
        )
