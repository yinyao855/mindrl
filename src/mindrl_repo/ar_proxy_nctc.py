"""AR-scorer proxy for low-cost MINDRL dependence smoke tests."""

from __future__ import annotations

import math
from collections.abc import Sequence


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
