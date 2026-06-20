"""Toy flow/diffusion surrogate tradeoffs for MINDRL reproduction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowSurrogateSetting:
    reward: float
    surrogate_variance: float
    drift: float
    anchor_strength: float
    clip_range: float
    smoothness_weight: float = 0.0


@dataclass(frozen=True)
class FlowSurrogateResult:
    effective_update: float
    residual_drift: float
    risk_adjusted_objective: float


def evaluate_flow_surrogate(setting: FlowSurrogateSetting) -> FlowSurrogateResult:
    """Evaluate a scalar toy objective for anchored flow-style updates."""

    if setting.anchor_strength < 0:
        raise ValueError("anchor_strength must be non-negative")
    if setting.clip_range <= 0:
        raise ValueError("clip_range must be positive")
    if setting.surrogate_variance < 0 or setting.drift < 0:
        raise ValueError("variance and drift must be non-negative")

    effective_update = setting.reward * min(1.0, setting.clip_range / 0.2)
    residual_drift = setting.drift / (1.0 + setting.anchor_strength)
    risk = (
        setting.surrogate_variance * setting.clip_range
        + residual_drift
        + setting.smoothness_weight * residual_drift**2
    )
    return FlowSurrogateResult(
        effective_update=effective_update,
        residual_drift=residual_drift,
        risk_adjusted_objective=effective_update - risk,
    )


def sweep_anchor_strengths(
    reward: float,
    surrogate_variance: float,
    drift: float,
    anchors: list[float],
    clip_range: float,
) -> list[FlowSurrogateResult]:
    """Evaluate the same branch under several anchor strengths."""

    return [
        evaluate_flow_surrogate(
            FlowSurrogateSetting(
                reward=reward,
                surrogate_variance=surrogate_variance,
                drift=drift,
                anchor_strength=anchor,
                clip_range=clip_range,
            )
        )
        for anchor in anchors
    ]
