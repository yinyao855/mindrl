"""MVP smoke benchmarks for AR and diffusion tutorials."""

from __future__ import annotations

import json
from pathlib import Path

from mindrl.ar_training import compute_grpo_objective, exact_match_reward
from mindrl.core import AlgorithmConfig, RolloutBatch, RolloutSample, TrainReport
from mindrl.diffusion_training import (
    DiffusionTrajectory,
    compressibility_reward,
    summarize_diffusion_run,
)


def build_ar_grpo_smoke_report() -> TrainReport:
    """Run a deterministic AR GRPO smoke objective without model downloads."""

    batch = RolloutBatch(
        samples=(
            RolloutSample("math-a", "2+2", "4", "ar", {"prompt_id": "math", "answer": "4"}),
            RolloutSample("math-b", "2+2", "5", "ar", {"prompt_id": "math", "answer": "4"}),
            RolloutSample("code-a", "is even", "true", "ar", {"prompt_id": "code", "answer": "true"}),
            RolloutSample("code-b", "is even", "false", "ar", {"prompt_id": "code", "answer": "true"}),
        )
    )
    rewards = exact_match_reward(batch)
    objective = compute_grpo_objective(
        batch,
        rewards,
        logprob_ratios={sample_id: 1.0 for sample_id in batch.sample_ids},
        kl_by_sample={sample_id: 0.0 for sample_id in batch.sample_ids},
    )
    return TrainReport(
        run_name="ar-grpo-smoke",
        algorithm=AlgorithmConfig(
            name="grpo",
            branch="ar",
            hyperparameters={"group_size": 2, "model": "mock-ar"},
        ),
        metrics={
            "reward_mean": rewards.mean_reward,
            "objective": objective.objective,
            "policy_term": objective.diagnostics["policy_term"],
        },
        artifacts={"dataset": "builtin://ar-smoke"},
    )


def build_diffusion_ddpo_smoke_report() -> TrainReport:
    """Run a deterministic diffusion DDPO smoke objective without image models."""

    trajectories = (
        DiffusionTrajectory("img-a", "red cat", (-0.1, -0.2), anchor_distance=0.1),
        DiffusionTrajectory("img-b", "red cat", (-0.4, -0.3), anchor_distance=0.3),
    )
    rewards = compressibility_reward({"img-a": "aa", "img-b": "aaaaaaaa"})
    return summarize_diffusion_run("diffusion-ddpo-smoke", trajectories, rewards)


def write_reports(output_dir: Path, reports: tuple[TrainReport, ...]) -> dict[str, Path]:
    """Write JSONL and Markdown reports for smoke runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "mvp_smoke_reports.jsonl"
    markdown_path = output_dir / "mvp_smoke_report.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report.to_json_record(), sort_keys=True) + "\n")
    markdown = "\n\n".join(report.to_markdown() for report in reports)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return {"jsonl": jsonl_path, "markdown": markdown_path}
