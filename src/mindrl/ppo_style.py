"""Dependency-light PPO-style objective diagnostics for AR rollouts."""

from __future__ import annotations

from collections import defaultdict

from mindrl.core import ObjectiveOutput, RewardOutput, RolloutBatch


def compute_ppo_style_objective(
    batch: RolloutBatch,
    rewards: RewardOutput,
    logprob_ratios: dict[str, float],
    kl_by_sample: dict[str, float] | None = None,
    clip_range: float = 0.2,
    kl_weight: float = 0.0,
    prompt_id_key: str = "prompt_id",
) -> ObjectiveOutput:
    """Compute a small clipped PPO-style objective over grouped rollouts."""

    rewards.validate_for(batch)
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in batch.samples:
        prompt_id = str(sample.metadata.get(prompt_id_key, sample.prompt))
        groups[prompt_id].append(sample.sample_id)

    advantages: dict[str, float] = {}
    for sample_ids in groups.values():
        baseline = sum(rewards.sample_rewards[sid] for sid in sample_ids) / len(sample_ids)
        for sample_id in sample_ids:
            advantages[sample_id] = rewards.sample_rewards[sample_id] - baseline

    unclipped_terms: list[float] = []
    clipped_terms: list[float] = []
    for sample_id, advantage in advantages.items():
        ratio = logprob_ratios.get(sample_id, 1.0)
        clipped_ratio = min(1.0 + clip_range, max(1.0 - clip_range, ratio))
        unclipped = ratio * advantage
        clipped = clipped_ratio * advantage
        if advantage >= 0:
            clipped_terms.append(min(unclipped, clipped))
        else:
            clipped_terms.append(max(unclipped, clipped))
        unclipped_terms.append(unclipped)

    policy_term = sum(unclipped_terms) / max(1, len(unclipped_terms))
    clipped_policy_term = sum(clipped_terms) / max(1, len(clipped_terms))
    kl = 0.0
    if kl_by_sample:
        kl = sum(kl_by_sample.get(sample_id, 0.0) for sample_id in advantages) / max(
            1, len(advantages)
        )
    objective = -(clipped_policy_term - kl_weight * kl)
    return ObjectiveOutput(
        objective=objective,
        sample_weights=advantages,
        diagnostics={
            "reward_mean": rewards.mean_reward,
            "policy_term": policy_term,
            "clipped_policy_term": clipped_policy_term,
            "kl": kl,
        },
    )
