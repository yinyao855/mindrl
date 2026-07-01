"""GRPO rollout loop primitives for AR policies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from mindrl.ar_training import compute_grpo_objective
from mindrl.core import (
    AlgorithmConfig,
    ObjectiveOutput,
    RewardOutput,
    RolloutBatch,
    RolloutSample,
    TrainReport,
)


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for group-relative policy optimization smoke steps."""

    group_size: int = 4
    kl_weight: float = 0.0
    run_name: str = "ar-grpo-rollout-smoke"


@dataclass(frozen=True)
class GRPOStepResult:
    batch: RolloutBatch
    rewards: RewardOutput
    objective: ObjectiveOutput
    report: TrainReport


class GroupRolloutPolicy(Protocol):
    def rollout(self, prompts: tuple[str, ...], group_size: int) -> RolloutBatch:
        ...

    def logprob_ratios(self, batch: RolloutBatch) -> dict[str, float]:
        ...

    def kl(self, batch: RolloutBatch) -> dict[str, float]:
        ...


class RewardAdapter(Protocol):
    def score(self, batch: RolloutBatch) -> RewardOutput:
        ...


class MockGroupRolloutPolicy:
    """Deterministic grouped rollout policy for GRPO smoke tests."""

    def __init__(
        self,
        completions: dict[str, tuple[str, ...]],
        logprob_ratios: dict[str, float],
        kl_by_sample: dict[str, float] | None = None,
    ) -> None:
        self.completions = completions
        self._logprob_ratios = logprob_ratios
        self._kl_by_sample = kl_by_sample or {}

    def rollout(self, prompts: tuple[str, ...], group_size: int) -> RolloutBatch:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        samples: list[RolloutSample] = []
        for prompt_index, prompt in enumerate(prompts):
            choices = self.completions[prompt]
            if len(choices) < group_size:
                raise ValueError(f"not enough completions for prompt {prompt}")
            for group_index in range(group_size):
                samples.append(
                    RolloutSample(
                        sample_id=f"grpo-{prompt_index}-{group_index}",
                        prompt=prompt,
                        response=choices[group_index],
                        branch="ar",
                        metadata={"prompt_id": prompt, "group_index": group_index},
                    )
                )
        return RolloutBatch(samples=tuple(samples))

    def logprob_ratios(self, batch: RolloutBatch) -> dict[str, float]:
        return {sample_id: self._logprob_ratios[sample_id] for sample_id in batch.sample_ids}

    def kl(self, batch: RolloutBatch) -> dict[str, float]:
        return {sample_id: self._kl_by_sample.get(sample_id, 0.0) for sample_id in batch.sample_ids}


class ExactAnswerRewardAdapter:
    """Exact-match reward adapter keyed by prompt id."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = {prompt: answer.strip().lower() for prompt, answer in answers.items()}

    def score(self, batch: RolloutBatch) -> RewardOutput:
        rewards: dict[str, float] = {}
        for sample in batch.samples:
            prompt_id = str(sample.metadata["prompt_id"])
            expected = self.answers[prompt_id]
            rewards[sample.sample_id] = 1.0 if sample.response.strip().lower() == expected else 0.0
        return RewardOutput(rewards)


class NumericAnswerRewardAdapter:
    """Reward the first numeric answer in a response."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = {prompt: self._first_number(answer) for prompt, answer in answers.items()}

    def score(self, batch: RolloutBatch) -> RewardOutput:
        rewards: dict[str, float] = {}
        for sample in batch.samples:
            prompt_id = str(sample.metadata["prompt_id"])
            expected = self.answers[prompt_id]
            actual = self._first_number(sample.response)
            rewards[sample.sample_id] = 1.0 if actual == expected else 0.0
        return RewardOutput(rewards)

    @staticmethod
    def _first_number(text: str) -> str:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return match.group(0) if match else ""


class StrictNumericAnswerRewardAdapter:
    """Reward only responses that contain exactly one numeric answer."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = {prompt: self._normalize(answer) for prompt, answer in answers.items()}

    def score(self, batch: RolloutBatch) -> RewardOutput:
        rewards: dict[str, float] = {}
        metadata: dict[str, dict[str, str]] = {}
        for sample in batch.samples:
            prompt_id = str(sample.metadata["prompt_id"])
            expected = self.answers[prompt_id]
            actual = self._normalize(sample.response)
            rewards[sample.sample_id] = 1.0 if actual == expected else 0.0
            metadata[sample.sample_id] = {"expected": expected, "actual": actual}
        return RewardOutput(rewards, metadata)

    @staticmethod
    def _normalize(text: str) -> str:
        stripped = text.strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            return ""
        return stripped


def run_grpo_step(
    prompts: tuple[str, ...],
    policy: GroupRolloutPolicy,
    reward_adapter: RewardAdapter,
    config: GRPOConfig | None = None,
) -> GRPOStepResult:
    """Run one GRPO step over grouped student rollouts."""

    config = config or GRPOConfig()
    batch = policy.rollout(prompts, config.group_size)
    rewards = reward_adapter.score(batch)
    objective = compute_grpo_objective(
        batch,
        rewards,
        logprob_ratios=policy.logprob_ratios(batch),
        kl_by_sample=policy.kl(batch),
        kl_weight=config.kl_weight,
    )
    report = TrainReport(
        run_name=config.run_name,
        algorithm=AlgorithmConfig(
            name="grpo",
            branch="ar",
            hyperparameters={"group_size": config.group_size, "kl_weight": config.kl_weight},
        ),
        metrics={
            "objective": objective.objective,
            "reward_mean": rewards.mean_reward,
            "group_size": float(config.group_size),
            "kl": objective.diagnostics["kl"],
            "policy_term": objective.diagnostics["policy_term"],
        },
    )
    return GRPOStepResult(batch=batch, rewards=rewards, objective=objective, report=report)
