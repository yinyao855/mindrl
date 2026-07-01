"""Generic teacher-guided on-policy update primitives.

The module keeps OPD branch-native: AR can use token logprob gaps, diffusion can
use denoising vectors, and flow can use velocity/surrogate vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from mindrl.core import AlgorithmConfig, ObjectiveOutput, RolloutBatch, TeacherSignal, TrainReport


@dataclass(frozen=True)
class OnPolicyState:
    """One state sampled or visited by the current student branch."""

    state_id: str
    branch: str
    payload: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherGuidance:
    """Teacher-provided local target for a student-visited state."""

    state_id: str
    branch: str
    target: tuple[float, ...]
    signal_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherGuidedObjectiveConfig:
    """Configuration for vector teacher-guided objectives."""

    name: str = "teacher_guided_opd"
    branch: str = "generic"
    per_element_clip: float | None = None


@dataclass(frozen=True)
class ToyTeacherGuidedUpdateResult:
    """Before/after result for dependency-light teacher-guided toy updates."""

    before: ObjectiveOutput
    after: ObjectiveOutput
    updated_states: tuple[OnPolicyState, ...]
    report: TrainReport


def rollout_batch_to_on_policy_states(
    batch: RolloutBatch,
    student_scores: dict[str, tuple[float, ...]],
) -> tuple[OnPolicyState, ...]:
    """Convert an AR rollout batch plus student scores to generic states."""

    states: list[OnPolicyState] = []
    for sample in batch.samples:
        if sample.sample_id not in student_scores:
            raise ValueError(f"missing student score for {sample.sample_id}")
        states.append(
            OnPolicyState(
                state_id=sample.sample_id,
                branch=sample.branch,
                payload=student_scores[sample.sample_id],
                metadata=dict(sample.metadata),
            )
        )
    return tuple(states)


def teacher_signals_to_guidance(
    teacher_signals: tuple[TeacherSignal, ...],
    branch: str = "ar",
) -> tuple[TeacherGuidance, ...]:
    """Convert token teacher signals to generic guidance objects."""

    return tuple(
        TeacherGuidance(
            state_id=signal.sample_id,
            branch=branch,
            target=signal.token_logprobs,
            signal_type="token_logprob",
            metadata={"topk_tokens": signal.topk_tokens},
        )
        for signal in teacher_signals
    )


def compute_teacher_guided_objective(
    states: tuple[OnPolicyState, ...],
    guidance: tuple[TeacherGuidance, ...],
    config: TeacherGuidedObjectiveConfig | None = None,
) -> ObjectiveOutput:
    """Compute a clipped vector gap objective over student-visited states."""

    config = config or TeacherGuidedObjectiveConfig()
    guidance_by_id = {item.state_id: item for item in guidance}
    sample_losses: dict[str, float] = {}
    raw_losses: list[float] = []
    clipped_losses: list[float] = []
    clipped_elements = 0

    for state in states:
        teacher = guidance_by_id.get(state.state_id)
        if teacher is None:
            raise ValueError(f"missing teacher guidance for {state.state_id}")
        if state.branch != teacher.branch:
            raise ValueError(f"branch mismatch for {state.state_id}: {state.branch} != {teacher.branch}")
        if len(state.payload) != len(teacher.target):
            raise ValueError(f"payload/target length mismatch for {state.state_id}")

        state_losses: list[float] = []
        for student_value, teacher_value in zip(state.payload, teacher.target):
            raw = abs(student_value - teacher_value)
            clipped = raw
            if config.per_element_clip is not None and raw > config.per_element_clip:
                clipped = config.per_element_clip
                clipped_elements += 1
            raw_losses.append(raw)
            clipped_losses.append(clipped)
            state_losses.append(clipped)
        sample_losses[state.state_id] = round(mean(state_losses), 12) if state_losses else 0.0

    objective = round(mean(clipped_losses), 12) if clipped_losses else 0.0
    raw_objective = round(mean(raw_losses), 12) if raw_losses else 0.0
    element_count = float(len(raw_losses))
    return ObjectiveOutput(
        objective=objective,
        sample_weights=sample_losses,
        diagnostics={
            "raw_objective": raw_objective,
            "elements": element_count,
            "clipped_elements": float(clipped_elements),
            "clipped_ratio": clipped_elements / element_count if element_count else 0.0,
        },
    )


def run_toy_teacher_guided_update(
    states: tuple[OnPolicyState, ...],
    guidance: tuple[TeacherGuidance, ...],
    config: TeacherGuidedObjectiveConfig | None = None,
    learning_rate: float = 0.1,
) -> ToyTeacherGuidedUpdateResult:
    """Move vector states one step toward teacher targets and report loss change."""

    if learning_rate < 0.0 or learning_rate > 1.0:
        raise ValueError("learning_rate must be between 0 and 1")
    config = config or TeacherGuidedObjectiveConfig()
    before = compute_teacher_guided_objective(states, guidance, config)
    guidance_by_id = {item.state_id: item for item in guidance}
    updated: list[OnPolicyState] = []
    for state in states:
        teacher = guidance_by_id.get(state.state_id)
        if teacher is None:
            raise ValueError(f"missing teacher guidance for {state.state_id}")
        if len(state.payload) != len(teacher.target):
            raise ValueError(f"payload/target length mismatch for {state.state_id}")
        new_payload = tuple(
            student_value + learning_rate * (teacher_value - student_value)
            for student_value, teacher_value in zip(state.payload, teacher.target)
        )
        updated.append(
            OnPolicyState(
                state_id=state.state_id,
                branch=state.branch,
                payload=new_payload,
                metadata=dict(state.metadata),
            )
        )
    updated_states = tuple(updated)
    after = compute_teacher_guided_objective(updated_states, guidance, config)
    report = TrainReport(
        run_name=f"{config.name}-toy-update",
        algorithm=AlgorithmConfig(
            name="teacher_guided_opd",
            branch=config.branch,
            hyperparameters={
                "learning_rate": learning_rate,
                "per_element_clip": config.per_element_clip,
            },
        ),
        metrics={
            "before_loss": before.objective,
            "after_loss": after.objective,
            "loss_delta": after.objective - before.objective,
            "clipped_ratio": after.diagnostics["clipped_ratio"],
        },
    )
    return ToyTeacherGuidedUpdateResult(before, after, updated_states, report)
