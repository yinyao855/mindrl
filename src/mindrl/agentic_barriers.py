"""Toy barrier metrics for interface-native agentic RL pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StepKind = Literal["reasoning", "tool_call", "browser_action", "memory_write", "delegate", "stop"]


@dataclass(frozen=True)
class AgentStep:
    kind: StepKind
    valid: bool = True
    latency: float = 0.0
    utility: float = 0.0
    irreversible: bool = False
    stale_observation: bool = False
    stop_should_have_happened_at: int | None = None


@dataclass(frozen=True)
class TraceBarrierSummary:
    tool_calls: int
    invalid_tool_calls: int
    mean_tool_latency: float
    stale_observation_steps: int
    irreversible_actions: int
    redundant_delegations: int
    premature_stop_steps: int
    late_stop_steps: int


def summarize_trace_barriers(trace: list[AgentStep]) -> TraceBarrierSummary:
    """Summarize branch-specific failure signals from an agent trajectory."""
    tool_steps = [step for step in trace if step.kind == "tool_call"]
    delegate_steps = [step for step in trace if step.kind == "delegate"]
    stop_index = next((idx for idx, step in enumerate(trace) if step.kind == "stop"), None)
    stop_target = next(
        (step.stop_should_have_happened_at for step in trace if step.stop_should_have_happened_at is not None),
        None,
    )

    premature = 0
    late = 0
    if stop_index is not None and stop_target is not None:
        premature = max(0, stop_target - stop_index)
        late = max(0, stop_index - stop_target)

    return TraceBarrierSummary(
        tool_calls=len(tool_steps),
        invalid_tool_calls=sum(1 for step in tool_steps if not step.valid),
        mean_tool_latency=_mean([step.latency for step in tool_steps]),
        stale_observation_steps=sum(1 for step in trace if step.stale_observation),
        irreversible_actions=sum(1 for step in trace if step.irreversible),
        redundant_delegations=sum(1 for step in delegate_steps if step.utility <= 0.0),
        premature_stop_steps=premature,
        late_stop_steps=late,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
