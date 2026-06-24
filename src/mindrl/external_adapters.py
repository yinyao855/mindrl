"""Export MindRL plans to external RL framework configurations."""

from __future__ import annotations

from mindrl.ar_trainer import ARTrainerPlan
from mindrl.grpo import GRPOConfig
from mindrl.opd import OPDConfig


def export_trl_grpo_config(plan: ARTrainerPlan, grpo: GRPOConfig) -> dict[str, object]:
    """Map a MindRL AR plan to a TRL GRPOConfig-like dictionary."""

    return {
        "model_name_or_path": plan.model.name,
        "num_generations": grpo.group_size,
        "beta": grpo.kl_weight,
        "per_device_train_batch_size": plan.trainer.per_device_batch_size,
        "gradient_accumulation_steps": plan.trainer.gradient_accumulation_steps,
        "learning_rate": plan.trainer.learning_rate,
        "max_steps": plan.trainer.max_steps,
        "gradient_checkpointing": plan.trainer.gradient_checkpointing,
        "lora_r": plan.lora.rank,
        "lora_alpha": plan.lora.alpha,
        "lora_dropout": plan.lora.dropout,
        "target_modules": list(plan.lora.target_modules),
        "load_in_4bit": plan.lora.quantization == "4bit",
    }


def export_verl_opd_config(
    plan: ARTrainerPlan,
    opd: OPDConfig,
    teacher_model: str,
) -> dict[str, object]:
    """Map a MindRL OPD plan to a verl-style configuration dictionary."""

    return {
        "algorithm": "opd",
        "student_model": plan.model.name,
        "teacher_model": teacher_model,
        "rollout": {
            "max_steps": plan.trainer.max_steps,
            "micro_batch_size": plan.trainer.per_device_batch_size,
        },
        "distillation": {
            "per_token_clip": opd.per_token_clip,
            "track_entropy": True,
            "track_token_kl": True,
        },
        "lora": {
            "rank": plan.lora.rank,
            "alpha": plan.lora.alpha,
            "quantization": plan.lora.quantization,
        },
    }


def export_openrlhf_args(plan: ARTrainerPlan, grpo: GRPOConfig) -> str:
    """Return an OpenRLHF command argument string for GRPO-style runs."""

    args = [
        f"--pretrain {plan.model.name}",
        "--algo.advantage.estimator group_norm",
        f"--num_generation {grpo.group_size}",
        f"--kl_coef {grpo.kl_weight}",
        f"--micro_train_batch_size {plan.trainer.per_device_batch_size}",
        f"--gradient_accumulation_steps {plan.trainer.gradient_accumulation_steps}",
        f"--max_steps {plan.trainer.max_steps}",
        f"--learning_rate {plan.trainer.learning_rate}",
        f"--lora_rank {plan.lora.rank}",
        f"--lora_alpha {plan.lora.alpha}",
    ]
    if plan.lora.quantization == "4bit":
        args.append("--load_in_4bit")
    if plan.trainer.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    return " ".join(args)
