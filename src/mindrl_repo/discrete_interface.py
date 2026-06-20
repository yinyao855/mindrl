"""Discrete-side MINDRL utilities.

This module implements the lightweight pieces needed to reproduce the paper's
nCTC measurement and adaptive parallel decoding schedule without requiring a
specific language-model backend.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def sample_distance_controlled_block(
    eligible_positions: Sequence[int],
    block_size: int,
    gap: int,
    seed: int,
) -> list[int]:
    """Return up to ``block_size`` positions separated by at least ``gap`` tokens."""
    if block_size < 0:
        raise ValueError("block_size must be non-negative")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    candidates = list(eligible_positions)
    random.Random(seed).shuffle(candidates)

    block: list[int] = []
    for position in candidates:
        if all(abs(position - chosen) >= gap for chosen in block):
            block.append(position)
        if len(block) == block_size:
            break
    return block


def estimate_nctc_from_logprobs(
    joint_order_logprobs: Sequence[Sequence[float]],
    marginal_logprobs: Sequence[float],
) -> float:
    """Estimate pair-normalized nCTC from joint chain-rule and marginal logprobs.

    ``joint_order_logprobs`` contains K chain-rule orderings. Each ordering stores
    log p_ref(x_mk | C, x_m<k). ``marginal_logprobs`` stores log p_ref(x_i | C)
    for the same block under the unrevealed visibility context.
    """
    block_size = len(marginal_logprobs)
    if block_size < 2:
        return 0.0
    if not joint_order_logprobs:
        raise ValueError("joint_order_logprobs must contain at least one ordering")

    for ordering in joint_order_logprobs:
        if len(ordering) != block_size:
            raise ValueError("each joint ordering must match marginal block size")

    mean_joint = sum(sum(ordering) for ordering in joint_order_logprobs) / len(
        joint_order_logprobs
    )
    log_gap = mean_joint - sum(marginal_logprobs)
    return log_gap / math.comb(block_size, 2)


def adaptive_block_size(
    uncertainties: Sequence[float],
    alpha: float,
    b_min: int,
    b_max: int,
    eps: float = 1e-5,
) -> int:
    """Compute Eq. 21's inverse uncertainty schedule."""
    if not uncertainties:
        raise ValueError("uncertainties must not be empty")
    if b_min < 1:
        raise ValueError("b_min must be at least 1")
    if b_max < b_min:
        raise ValueError("b_max must be greater than or equal to b_min")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    mean_uncertainty = sum(uncertainties) / len(uncertainties)
    raw_size = math.floor(alpha / (mean_uncertainty + eps))
    return max(b_min, min(b_max, raw_size))
