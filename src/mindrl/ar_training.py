"""Transparent AR-LLM objectives for the MVP training tutorials."""

from __future__ import annotations

from collections import defaultdict

from mindrl.core import ObjectiveOutput, RewardOutput, RolloutBatch


def exact_match_reward(batch: RolloutBatch, answer_key: str = "answer") -> RewardOutput:
    """Score AR samples against exact answers stored in sample metadata."""

    rewards: dict[str, float] = {}
    metadata: dict[str, dict[str, str]] = {}
    for sample in batch.samples:
        expected = str(sample.metadata.get(answer_key, "")).strip().lower()
        actual = sample.response.strip().lower()
        rewards[sample.sample_id] = 1.0 if expected and actual == expected else 0.0
        metadata[sample.sample_id] = {"expected": expected, "actual": actual}
    return RewardOutput(sample_rewards=rewards, reward_metadata=metadata)


def compute_grpo_objective(
    batch: RolloutBatch,
    rewards: RewardOutput,
    logprob_ratios: dict[str, float],
    kl_by_sample: dict[str, float] | None = None,
    kl_weight: float = 0.0,
    prompt_id_key: str = "prompt_id",
) -> ObjectiveOutput:
    """Compute a small GRPO-style objective with group-relative advantages."""

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

    policy_term = sum(
        logprob_ratios.get(sample_id, 1.0) * advantage
        for sample_id, advantage in advantages.items()
    ) / max(1, len(advantages))
    kl_term = 0.0
    if kl_by_sample:
        kl_term = sum(kl_by_sample.get(sample_id, 0.0) for sample_id in advantages) / max(
            1, len(advantages)
        )
    objective = -(policy_term - kl_weight * kl_term)
    return ObjectiveOutput(
        objective=objective,
        sample_weights=advantages,
        diagnostics={
            "reward_mean": rewards.mean_reward,
            "policy_term": policy_term,
            "kl": kl_term,
        },
    )


def compute_opd_objective(
    student_logprobs: dict[str, tuple[float, ...]],
    teacher_signals,
) -> ObjectiveOutput:
    """Match student logprobs to teacher logprobs on student rollouts."""

    sample_losses: dict[str, float] = {}
    token_losses: list[float] = []
    for signal in teacher_signals:
        student = student_logprobs.get(signal.sample_id)
        if student is None:
            raise ValueError(f"missing student logprobs for {signal.sample_id}")
        if len(student) != signal.token_count:
            raise ValueError(f"token count mismatch for {signal.sample_id}")
        losses = [abs(s - t) for s, t in zip(student, signal.token_logprobs)]
        sample_losses[signal.sample_id] = round(sum(losses) / max(1, len(losses)), 12)
        token_losses.extend(losses)

    objective = round(sum(token_losses) / max(1, len(token_losses)), 12)
    return ObjectiveOutput(
        objective=objective,
        sample_weights=sample_losses,
        diagnostics={"tokens": float(len(token_losses))},
    )


def compute_reinforce_baseline_objective(
    rewards: RewardOutput,
    logprobs: dict[str, float],
) -> ObjectiveOutput:
    """Compute a REINFORCE-style objective with the batch mean reward baseline."""

    baseline = rewards.mean_reward
    advantages = {
        sample_id: reward - baseline for sample_id, reward in rewards.sample_rewards.items()
    }
    policy_term = sum(
        logprobs.get(sample_id, 0.0) * advantage
        for sample_id, advantage in advantages.items()
    ) / max(1, len(advantages))
    return ObjectiveOutput(
        objective=-policy_term,
        sample_weights=advantages,
        diagnostics={"baseline": baseline, "policy_term": policy_term},
    )
