"""Core framework objects shared by AR and diffusion MVP pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class RolloutSample:
    """One generated unit from any policy branch."""

    sample_id: str
    prompt: str
    response: str
    branch: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutBatch:
    """A batch of rollout samples with stable sample identifiers."""

    samples: tuple[RolloutSample, ...]

    def __post_init__(self) -> None:
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample_id in rollout batch")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    def by_branch(self, branch: str) -> tuple[RolloutSample, ...]:
        return tuple(sample for sample in self.samples if sample.branch == branch)


@dataclass(frozen=True)
class RewardOutput:
    """Reward values keyed by rollout sample id."""

    sample_rewards: dict[str, float]
    reward_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        if not self.sample_rewards:
            return 0.0
        return mean(self.sample_rewards.values())

    def validate_for(self, batch: RolloutBatch) -> None:
        expected = set(batch.sample_ids)
        actual = set(self.sample_rewards)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"reward sample ids mismatch: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class TeacherSignal:
    """Dense teacher supervision for one on-policy rollout."""

    sample_id: str
    token_logprobs: tuple[float, ...]
    topk_tokens: tuple[tuple[str, ...], ...] = ()

    @property
    def token_count(self) -> int:
        return len(self.token_logprobs)

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return mean(self.token_logprobs)


@dataclass(frozen=True)
class AlgorithmConfig:
    """Serializable algorithm configuration used in reports."""

    name: str
    branch: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveOutput:
    """Loss-like scalar plus diagnostics emitted by an objective."""

    objective: float
    sample_weights: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainReport:
    """Small, serializable report for smoke runs and tutorials."""

    run_name: str
    algorithm: AlgorithmConfig
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "algorithm": asdict(self.algorithm),
            "metrics": dict(sorted(self.metrics.items())),
            "artifacts": dict(sorted(self.artifacts.items())),
        }

    def to_markdown(self) -> str:
        metric_lines = "\n".join(
            f"- `{name}`: {value}" for name, value in sorted(self.metrics.items())
        )
        artifact_lines = "\n".join(
            f"- `{name}`: `{path}`" for name, path in sorted(self.artifacts.items())
        )
        if not metric_lines:
            metric_lines = "- No metrics recorded"
        if not artifact_lines:
            artifact_lines = "- No artifacts recorded"
        return (
            f"# {self.run_name}\n\n"
            f"Algorithm: `{self.algorithm.name}` on `{self.algorithm.branch}`\n\n"
            "## Metrics\n"
            f"{metric_lines}\n\n"
            "## Artifacts\n"
            f"{artifact_lines}\n"
        )
