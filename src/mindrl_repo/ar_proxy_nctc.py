"""AR-scorer proxy for low-cost MINDRL dependence smoke tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence


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
    if block_size < 2:
        return 0.0

    log_gap = sum(joint_chain_logprobs) - sum(prompt_only_marginal_logprobs)
    return log_gap / math.comb(block_size, 2)


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
