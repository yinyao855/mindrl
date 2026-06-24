"""MVP smoke benchmarks for AR and diffusion tutorials."""

from __future__ import annotations

import json
from pathlib import Path

from mindrl.ar_trainer import qwen_lora_preset, summarize_ar_trainer_plan
from mindrl.controller_ablation import BranchScenario, summarize_controller_ablation
from mindrl.core import TrainReport
from mindrl.diffusion_adapter import (
    MockDiffusionPipeline,
    build_image_grid_manifest,
    collect_diffusion_rollouts,
)
from mindrl.diffusion_training import (
    compressibility_reward,
    summarize_diffusion_run,
)
from mindrl.grpo import ExactAnswerRewardAdapter, GRPOConfig, MockGroupRolloutPolicy, run_grpo_step
from mindrl.interface_controller import BarrierProfile, PolicySpec
from mindrl.opd import MappingTeacherSignalAdapter, MockARPolicy, OPDConfig, run_opd_step


def build_ar_grpo_smoke_report() -> TrainReport:
    """Run a deterministic AR GRPO smoke objective without model downloads."""

    policy = MockGroupRolloutPolicy(
        completions={
            "math": ("4", "5"),
            "code": ("true", "false"),
        },
        logprob_ratios={
            "grpo-0-0": 1.0,
            "grpo-0-1": 1.0,
            "grpo-1-0": 1.0,
            "grpo-1-1": 1.0,
        },
    )
    return run_grpo_step(
        ("math", "code"),
        policy,
        ExactAnswerRewardAdapter({"math": "4", "code": "true"}),
        GRPOConfig(group_size=2, run_name="ar-grpo-smoke"),
    ).report


def build_ar_opd_smoke_report() -> TrainReport:
    """Run a deterministic OPD smoke step with clipping diagnostics."""

    student = MockARPolicy(
        responses={"math": "wait add four"},
        token_logprobs={"opd-0": (-1.3, -0.4, -0.3)},
    )
    teacher = MappingTeacherSignalAdapter(
        token_logprobs={"opd-0": (-0.1, -0.2, -0.2)},
        topk_tokens={"opd-0": (("ok", "wait"), ("add",), ("four",))},
        entropies={"opd-0": (1.5, 0.4, 0.3)},
    )
    return run_opd_step(
        ("math",),
        student,
        teacher,
        OPDConfig(per_token_clip=0.25),
    ).report


def build_ar_lora_plan_report(scale: str = "qwen-7b") -> TrainReport:
    """Build a reportable AR LoRA/QLoRA training plan."""

    return summarize_ar_trainer_plan(
        f"ar-lora-plan-{scale}",
        qwen_lora_preset(scale),
    )


def build_diffusion_ddpo_smoke_report() -> TrainReport:
    """Run a deterministic diffusion DDPO smoke objective without image models."""

    pipeline = MockDiffusionPipeline(
        images={"red cat": "aa", "blue dog": "aaaaaaaa"},
        captions={"red cat": "a red cat", "blue dog": "a blue dog"},
        step_logprobs={"red cat": (-0.1, -0.2), "blue dog": (-0.4, -0.3)},
        anchor_distances={"red cat": 0.1, "blue dog": 0.3},
    )
    batch = collect_diffusion_rollouts(("red cat", "blue dog"), pipeline)
    rewards = compressibility_reward(batch.images)
    report = summarize_diffusion_run("diffusion-ddpo-smoke", batch.trajectories, rewards)
    artifacts = dict(report.artifacts)
    artifacts["image_grid_manifest"] = build_image_grid_manifest(batch)
    return TrainReport(
        run_name=report.run_name,
        algorithm=report.algorithm,
        metrics=report.metrics,
        artifacts=artifacts,
    )


def build_controller_ablation_report() -> TrainReport:
    """Run a mixed AR/diffusion controller ablation smoke benchmark."""

    scenarios = [
        BranchScenario(
            spec=PolicySpec(
                name="ar_reasoning",
                modality="text",
                structure="ar",
                score_availability="exact",
                default_granularity=1,
            ),
            profile=BarrierProfile(),
            reward=1.0,
        ),
        BranchScenario(
            spec=PolicySpec(
                name="diffusion_image",
                modality="image",
                structure="diffusion",
                score_availability="surrogate",
                default_granularity=8,
            ),
            profile=BarrierProfile(surrogate_variance=0.4, drift=0.5, smoothness=0.1),
            reward=0.8,
        ),
    ]
    return summarize_controller_ablation("controller-ablation-smoke", scenarios)


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
