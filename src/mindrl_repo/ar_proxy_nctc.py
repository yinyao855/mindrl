"""AR-scorer proxy for low-cost MINDRL dependence smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from mindrl_repo.discrete_interface import estimate_nctc_from_logprobs


@dataclass(frozen=True)
class NCTCProxyRecord:
    """One scored prompt/completion block."""

    task: str
    dependency_group: str
    token_count: int
    joint_logprob: float
    marginal_logprob: float
    pair_normalized_gap: float


@dataclass(frozen=True)
class NCTCProxySummary:
    """Mean nCTC proxy for a group of records."""

    group: str
    count: int
    mean_pair_normalized_gap: float


@dataclass(frozen=True)
class BlockNCTCRecord:
    """One sampled completion block scored with one or more chain-rule orders."""

    task: str
    dependency_group: str
    block_start: int
    token_count: int
    order_count: int
    mean_joint_logprob: float
    marginal_logprob: float
    pair_normalized_nctc: float


@dataclass(frozen=True)
class BootstrapCI:
    """Deterministic bootstrap interval for a scalar record field."""

    mean: float
    low: float
    high: float
    samples: int


def pair_normalized_dependency_gap(
    joint_chain_logprobs: Sequence[float],
    prompt_only_marginal_logprobs: Sequence[float],
) -> float:
    """Compute a pair-normalized dependency gap from AR logprob terms.

    This is a smoke-test proxy for nCTC when only a causal LM is available. The
    joint term uses AR chain logprobs over a completion block, while marginals
    score each block token as a one-token continuation from the same prompt.
    """
    block_size = len(joint_chain_logprobs)
    if block_size != len(prompt_only_marginal_logprobs):
        raise ValueError("joint and marginal logprob lists must have the same length")
    return estimate_nctc_from_logprobs(
        [joint_chain_logprobs],
        prompt_only_marginal_logprobs,
    )


def build_proxy_record(
    task: str,
    dependency_group: str,
    joint_chain_logprobs: Sequence[float],
    prompt_only_marginal_logprobs: Sequence[float],
) -> NCTCProxyRecord:
    """Build a serializable record from scorer logprob terms."""

    return NCTCProxyRecord(
        task=task,
        dependency_group=dependency_group,
        token_count=len(joint_chain_logprobs),
        joint_logprob=sum(joint_chain_logprobs),
        marginal_logprob=sum(prompt_only_marginal_logprobs),
        pair_normalized_gap=pair_normalized_dependency_gap(
            joint_chain_logprobs,
            prompt_only_marginal_logprobs,
        ),
    )


def build_block_nctc_record(
    task: str,
    dependency_group: str,
    block_start: int,
    joint_order_logprobs: Sequence[Sequence[float]],
    marginal_logprobs: Sequence[float],
) -> BlockNCTCRecord:
    """Build a block-level multi-order nCTC record.

    This is closer to the paper protocol than the legacy whole-completion AR
    proxy because the unit of analysis is a sampled block and the joint term can
    average multiple chain-rule orders.
    """

    if not joint_order_logprobs:
        raise ValueError("joint_order_logprobs must contain at least one ordering")
    token_count = len(marginal_logprobs)
    if token_count == 0:
        raise ValueError("marginal_logprobs must not be empty")
    for ordering in joint_order_logprobs:
        if len(ordering) != token_count:
            raise ValueError("each joint ordering must match marginal block size")

    mean_joint = sum(sum(ordering) for ordering in joint_order_logprobs) / len(
        joint_order_logprobs
    )
    marginal = sum(marginal_logprobs)
    return BlockNCTCRecord(
        task=task,
        dependency_group=dependency_group,
        block_start=block_start,
        token_count=token_count,
        order_count=len(joint_order_logprobs),
        mean_joint_logprob=mean_joint,
        marginal_logprob=marginal,
        pair_normalized_nctc=estimate_nctc_from_logprobs(
            joint_order_logprobs,
            marginal_logprobs,
        ),
    )


def default_block_starts(
    token_count: int,
    block_size: int,
    stride: int | None = None,
) -> list[int]:
    """Return deterministic start offsets for benchmark block sampling."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if stride is None:
        stride = block_size
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if token_count < block_size:
        return []
    return list(range(0, token_count - block_size + 1, stride))


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Compute a deterministic percentile bootstrap CI for a mean."""

    if not values:
        raise ValueError("values must not be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    import random

    rng = random.Random(seed)
    values_list = list(values)
    draws: list[float] = []
    for _ in range(samples):
        sample = [rng.choice(values_list) for _ in values_list]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low_index = max(0, int((alpha / 2) * samples))
    high_index = min(samples - 1, int((1 - alpha / 2) * samples))
    return BootstrapCI(
        mean=sum(values_list) / len(values_list),
        low=draws[low_index],
        high=draws[high_index],
        samples=samples,
    )


def summarize_proxy_records(
    records: Sequence[NCTCProxyRecord],
    group_by: str,
) -> list[NCTCProxySummary]:
    """Summarize records by ``task`` or ``dependency_group``."""

    if group_by not in {"task", "dependency_group"}:
        raise ValueError("group_by must be 'task' or 'dependency_group'")

    grouped: dict[str, list[NCTCProxyRecord]] = {}
    for record in records:
        key = record.task if group_by == "task" else record.dependency_group
        grouped.setdefault(key, []).append(record)

    summaries: list[NCTCProxySummary] = []
    for group, group_records in sorted(grouped.items()):
        mean_gap = sum(record.pair_normalized_gap for record in group_records) / len(
            group_records
        )
        summaries.append(
            NCTCProxySummary(
                group=group,
                count=len(group_records),
                mean_pair_normalized_gap=mean_gap,
            )
        )
    return summaries
