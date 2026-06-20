"""Synthetic barrier checks for MINDRL's discrete factorization argument."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

JointDistribution = Mapping[tuple[int, ...], float]


def binary_pair_distribution(diagonal_cell_prob: float) -> dict[tuple[int, int], float]:
    """Create a symmetric binary pair distribution with tunable dependence.

    ``diagonal_cell_prob=0.25`` is uniform and independent. Larger values put
    more mass on 00 and 11, increasing total correlation.
    """
    if not 0 <= diagonal_cell_prob <= 0.5:
        raise ValueError("diagonal_cell_prob must be between 0 and 0.5")
    off_diagonal = 0.5 - diagonal_cell_prob
    return {
        (0, 0): diagonal_cell_prob,
        (1, 1): diagonal_cell_prob,
        (0, 1): off_diagonal,
        (1, 0): off_diagonal,
    }


def total_correlation(joint: JointDistribution) -> float:
    """Compute CTC without conditioning for a finite joint distribution."""
    _validate_joint(joint)
    arity = len(next(iter(joint)))
    marginals = [_marginal(joint, (idx,)) for idx in range(arity)]

    tc = 0.0
    for outcome, prob in joint.items():
        if prob == 0:
            continue
        product = 1.0
        for idx, marginal in enumerate(marginals):
            product *= marginal[(outcome[idx],)]
        tc += prob * math.log(prob / product)
    return tc


def within_block_barrier(
    joint: JointDistribution,
    blocks: Iterable[Sequence[int]],
) -> float:
    """Sum total correlation over each proposed parallel decision block."""
    _validate_joint(joint)
    barrier = 0.0
    for block in blocks:
        if len(block) <= 1:
            continue
        barrier += total_correlation(_marginal(joint, tuple(block)))
    return barrier


def _marginal(joint: JointDistribution, indices: tuple[int, ...]) -> dict[tuple[int, ...], float]:
    marginal: dict[tuple[int, ...], float] = defaultdict(float)
    for outcome, prob in joint.items():
        key = tuple(outcome[idx] for idx in indices)
        marginal[key] += prob
    return dict(marginal)


def _validate_joint(joint: JointDistribution) -> None:
    if not joint:
        raise ValueError("joint distribution must not be empty")
    arities = {len(outcome) for outcome in joint}
    if len(arities) != 1:
        raise ValueError("all outcomes must have the same arity")
    total = sum(joint.values())
    if any(prob < 0 for prob in joint.values()):
        raise ValueError("probabilities must be non-negative")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to 1")
