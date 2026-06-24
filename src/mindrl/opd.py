"""On-policy distillation loop primitives.

The real model integration point is intentionally narrow: student policies
produce rollouts, teacher adapters score those student states, and the objective
applies token-level clipping/diagnostics before a trainer consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from mindrl.core import (
    AlgorithmConfig,
    ObjectiveOutput,
    RolloutBatch,
    RolloutSample,
    TeacherSignal,
    TrainReport,
)


@dataclass(frozen=True)
class OPDConfig:
    """Configuration for token-level OPD supervision."""

    per_token_clip: float | None = None
    run_name: str = "ar-opd-smoke"


@dataclass(frozen=True)
class OPDStepResult:
    """One on-policy distillation step with reportable diagnostics."""

    batch: RolloutBatch
    teacher_signals: tuple[TeacherSignal, ...]
    objective: ObjectiveOutput
    report: TrainReport


class ARPolicy(Protocol):
    """Student policy interface needed for OPD."""

    def rollout(self, prompts: tuple[str, ...]) -> RolloutBatch:
        ...

    def score(self, batch: RolloutBatch) -> dict[str, tuple[float, ...]]:
        ...


class TeacherSignalAdapter(Protocol):
    """Teacher scoring interface over student-generated rollout states."""

    def signals_for(self, batch: RolloutBatch) -> tuple[TeacherSignal, ...]:
        ...


class MockARPolicy:
    """Deterministic student policy for smoke tests and tutorials."""

    def __init__(
        self,
        responses: dict[str, str],
        token_logprobs: dict[str, tuple[float, ...]],
    ) -> None:
        self.responses = responses
        self.token_logprobs = token_logprobs

    def rollout(self, prompts: tuple[str, ...]) -> RolloutBatch:
        samples = []
        for index, prompt in enumerate(prompts):
            sample_id = f"opd-{index}"
            samples.append(
                RolloutSample(
                    sample_id=sample_id,
                    prompt=prompt,
                    response=self.responses[prompt],
                    branch="ar",
                    metadata={"prompt_id": prompt},
                )
            )
        return RolloutBatch(samples=tuple(samples))

    def score(self, batch: RolloutBatch) -> dict[str, tuple[float, ...]]:
        scores: dict[str, tuple[float, ...]] = {}
        for sample in batch.samples:
            if sample.sample_id not in self.token_logprobs:
                raise ValueError(f"missing student logprobs for {sample.sample_id}")
            scores[sample.sample_id] = self.token_logprobs[sample.sample_id]
        return scores


class MappingTeacherSignalAdapter:
    """Deterministic teacher adapter keyed by rollout sample id."""

    def __init__(
        self,
        token_logprobs: dict[str, tuple[float, ...]],
        topk_tokens: dict[str, tuple[tuple[str, ...], ...]] | None = None,
        entropies: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.token_logprobs = token_logprobs
        self.topk_tokens = topk_tokens or {}
        self.entropies = entropies or {}

    def signals_for(self, batch: RolloutBatch) -> tuple[TeacherSignal, ...]:
        signals: list[TeacherSignal] = []
        for sample in batch.samples:
            if sample.sample_id not in self.token_logprobs:
                raise ValueError(f"missing teacher logprobs for {sample.sample_id}")
            signals.append(
                TeacherSignal(
                    sample_id=sample.sample_id,
                    token_logprobs=self.token_logprobs[sample.sample_id],
                    topk_tokens=self.topk_tokens.get(sample.sample_id, ()),
                )
            )
        return tuple(signals)


def compute_clipped_opd_objective(
    student_logprobs: dict[str, tuple[float, ...]],
    teacher_signals: tuple[TeacherSignal, ...],
    config: OPDConfig | None = None,
    teacher_entropies: dict[str, tuple[float, ...]] | None = None,
) -> ObjectiveOutput:
    """Compute OPD loss with optional per-token clipping.

    The absolute logprob gap is a dependency-light proxy for token-level
    teacher/student divergence. Clipping keeps high-KL style or pivot tokens from
    dominating the update, matching the practical concern highlighted in OPD
    analyses.
    """

    config = config or OPDConfig()
    teacher_entropies = teacher_entropies or {}
    sample_losses: dict[str, float] = {}
    raw_losses: list[float] = []
    clipped_losses: list[float] = []
    entropy_values: list[float] = []
    clipped_tokens = 0

    for signal in teacher_signals:
        student = student_logprobs.get(signal.sample_id)
        if student is None:
            raise ValueError(f"missing student logprobs for {signal.sample_id}")
        if len(student) != signal.token_count:
            raise ValueError(f"token count mismatch for {signal.sample_id}")

        token_losses: list[float] = []
        for student_logprob, teacher_logprob in zip(student, signal.token_logprobs):
            raw = abs(student_logprob - teacher_logprob)
            clipped = raw
            if config.per_token_clip is not None and raw > config.per_token_clip:
                clipped = config.per_token_clip
                clipped_tokens += 1
            raw_losses.append(raw)
            clipped_losses.append(clipped)
            token_losses.append(clipped)

        if signal.sample_id in teacher_entropies:
            entropies = teacher_entropies[signal.sample_id]
            if len(entropies) != signal.token_count:
                raise ValueError(f"entropy count mismatch for {signal.sample_id}")
            entropy_values.extend(entropies)

        sample_losses[signal.sample_id] = round(mean(token_losses), 12)

    objective = round(mean(clipped_losses), 12) if clipped_losses else 0.0
    raw_objective = round(mean(raw_losses), 12) if raw_losses else 0.0
    mean_entropy = round(mean(entropy_values), 12) if entropy_values else 0.0
    return ObjectiveOutput(
        objective=objective,
        sample_weights=sample_losses,
        diagnostics={
            "raw_objective": raw_objective,
            "tokens": float(len(clipped_losses)),
            "clipped_tokens": float(clipped_tokens),
            "mean_teacher_entropy": mean_entropy,
        },
    )


def run_opd_step(
    prompts: tuple[str, ...],
    student_policy: ARPolicy,
    teacher_adapter: TeacherSignalAdapter,
    config: OPDConfig | None = None,
) -> OPDStepResult:
    """Run one dependency-light OPD step over student-generated states."""

    config = config or OPDConfig()
    batch = student_policy.rollout(prompts)
    student_logprobs = student_policy.score(batch)
    teacher_signals = teacher_adapter.signals_for(batch)
    teacher_entropies = getattr(teacher_adapter, "entropies", {})
    objective = compute_clipped_opd_objective(
        student_logprobs,
        teacher_signals,
        config=config,
        teacher_entropies=teacher_entropies,
    )
    report = TrainReport(
        run_name=config.run_name,
        algorithm=AlgorithmConfig(
            name="opd",
            branch="ar",
            hyperparameters={"per_token_clip": config.per_token_clip},
        ),
        metrics={
            "objective": objective.objective,
            "raw_objective": objective.diagnostics["raw_objective"],
            "tokens": objective.diagnostics["tokens"],
            "clipped_tokens": objective.diagnostics["clipped_tokens"],
            "mean_teacher_entropy": objective.diagnostics["mean_teacher_entropy"],
        },
    )
    return OPDStepResult(
        batch=batch,
        teacher_signals=teacher_signals,
        objective=objective,
        report=report,
    )
