"""DDPO-style diffusion MVP primitives.

The module keeps the first MVP dependency-light: real diffusers pipelines can
populate these trajectories, while tests and tutorials can use strings.
"""

from __future__ import annotations

from dataclasses import dataclass

from mindrl.core import AlgorithmConfig, ObjectiveOutput, RewardOutput, TrainReport
from mindrl.flow_diffusion_interface import FlowDiffusionTrace, evaluate_flow_adapter


@dataclass(frozen=True)
class DiffusionTrajectory:
    """One denoising trajectory sampled from a diffusion policy."""

    sample_id: str
    prompt: str
    step_logprobs: tuple[float, ...]
    anchor_distance: float = 0.0
    image_caption: str = ""


def compressibility_reward(images: dict[str, str]) -> RewardOutput:
    """Reward shorter serialized images as a deterministic compressibility proxy."""

    if not images:
        return RewardOutput({})
    max_len = max(len(payload) for payload in images.values()) or 1
    rewards = {
        sample_id: 1.0 - (len(payload) / max_len)
        for sample_id, payload in images.items()
    }
    return RewardOutput(sample_rewards=rewards)


def clip_alignment_reward(
    prompts: dict[str, str],
    image_captions: dict[str, str],
) -> RewardOutput:
    """Small CLIP-like proxy based on prompt/caption token overlap."""

    rewards: dict[str, float] = {}
    for sample_id, prompt in prompts.items():
        prompt_tokens = {token.lower() for token in prompt.split()}
        caption_tokens = {
            token.lower().strip(".,!?;:")
            for token in image_captions.get(sample_id, "").split()
        }
        if not prompt_tokens:
            rewards[sample_id] = 0.0
        else:
            rewards[sample_id] = len(prompt_tokens & caption_tokens) / len(prompt_tokens)
    return RewardOutput(sample_rewards=rewards)


def compute_ddpo_objective(
    trajectories: tuple[DiffusionTrajectory, ...],
    rewards: RewardOutput,
    kl_anchor_weight: float = 0.0,
) -> ObjectiveOutput:
    """Compute a DDPO-style score-function objective over denoising logprobs."""

    if not trajectories:
        return ObjectiveOutput(0.0, {}, {"reward_mean": 0.0, "anchor_penalty": 0.0})
    reward_mean = rewards.mean_reward
    advantages = {
        trajectory.sample_id: rewards.sample_rewards[trajectory.sample_id] - reward_mean
        for trajectory in trajectories
    }
    policy_term = sum(
        sum(trajectory.step_logprobs) * advantages[trajectory.sample_id]
        for trajectory in trajectories
    ) / len(trajectories)
    anchor_penalty = sum(t.anchor_distance for t in trajectories) / len(trajectories)
    objective = -(policy_term - kl_anchor_weight * anchor_penalty)
    return ObjectiveOutput(
        objective=objective,
        sample_weights=advantages,
        diagnostics={
            "reward_mean": reward_mean,
            "policy_term": policy_term,
            "anchor_penalty": anchor_penalty,
        },
    )


def summarize_diffusion_run(
    run_name: str,
    trajectories: tuple[DiffusionTrajectory, ...],
    rewards: RewardOutput,
) -> TrainReport:
    """Create a compact report and expose the controller adapter decision."""

    if not trajectories:
        return TrainReport(
            run_name=run_name,
            algorithm=AlgorithmConfig(name="ddpo", branch="diffusion"),
            metrics={"reward_mean": 0.0},
        )
    trace = FlowDiffusionTrace(
        reward=rewards.mean_reward,
        surrogate_scores=tuple(sum(t.step_logprobs) for t in trajectories),
        anchor_distances=tuple(t.anchor_distance for t in trajectories),
    )
    evaluation = evaluate_flow_adapter(trace)
    objective = compute_ddpo_objective(trajectories, rewards)
    return TrainReport(
        run_name=run_name,
        algorithm=AlgorithmConfig(name="ddpo", branch="diffusion"),
        metrics={
            "reward_mean": rewards.mean_reward,
            "objective": objective.objective,
            "anchor_penalty": objective.diagnostics["anchor_penalty"],
            "branch_weight": evaluation.decision.branch_weight,
        },
        artifacts={"adapter": evaluation.decision.adapter},
    )
