"""AR LoRA/QLoRA trainer planning primitives.

This module does not launch heavyweight training. It produces validated,
serializable plans that real PEFT/TRL/verl adapters can execute later.
"""

from __future__ import annotations

from dataclasses import dataclass

from mindrl.core import AlgorithmConfig, TrainReport


@dataclass(frozen=True)
class ARModelConfig:
    """Model identity and approximate scale."""

    name: str
    parameter_count_b: float
    dtype: str = "bf16"


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA or QLoRA adapter configuration."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    quantization: str | None = None


@dataclass(frozen=True)
class TrainerConfig:
    """Single-node trainer knobs used by MVP smoke planning."""

    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100
    learning_rate: float = 1e-5
    gradient_checkpointing: bool = True
    checkpoint_every: int = 50


@dataclass(frozen=True)
class ARTrainerPlan:
    """Validated AR trainer plan with rough resource estimates."""

    model: ARModelConfig
    lora: LoRAConfig
    trainer: TrainerConfig
    effective_batch_size: int
    estimated_vram_gb: float
    tags: tuple[str, ...]


def build_ar_trainer_plan(
    model: ARModelConfig,
    lora: LoRAConfig,
    trainer: TrainerConfig,
) -> ARTrainerPlan:
    """Build a deterministic LoRA/QLoRA plan and rough memory estimate."""

    if model.parameter_count_b <= 0:
        raise ValueError("parameter_count_b must be positive")
    if lora.rank < 1:
        raise ValueError("LoRA rank must be positive")
    if trainer.per_device_batch_size < 1:
        raise ValueError("per_device_batch_size must be positive")
    if trainer.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    bytes_per_param = 2.0 if model.dtype in {"bf16", "fp16"} else 4.0
    if lora.quantization == "4bit":
        bytes_per_param = 0.6
    base_memory = model.parameter_count_b * 1_000_000_000 * bytes_per_param / 1_000_000_000
    adapter_memory = model.parameter_count_b * lora.rank * 0.015
    activation_multiplier = 0.35 if trainer.gradient_checkpointing else 0.60
    activation_memory = model.parameter_count_b * trainer.per_device_batch_size * activation_multiplier
    estimated_vram = round(base_memory + adapter_memory + activation_memory + 1.5, 2)

    tags = ["lora"]
    if lora.quantization:
        tags.append("qlora")
    if trainer.gradient_checkpointing:
        tags.append("gradient_checkpointing")
    return ARTrainerPlan(
        model=model,
        lora=lora,
        trainer=trainer,
        effective_batch_size=trainer.per_device_batch_size
        * trainer.gradient_accumulation_steps,
        estimated_vram_gb=estimated_vram,
        tags=tuple(tags),
    )


def qwen_lora_preset(scale: str) -> ARTrainerPlan:
    """Return conservative Qwen LoRA/QLoRA presets for smoke planning."""

    normalized = scale.lower()
    presets = {
        "qwen-0.5b": ("Qwen/Qwen2.5-0.5B", 0.5, None, 2, 4),
        "qwen-1.5b": ("Qwen/Qwen2.5-1.5B", 1.5, "4bit", 1, 8),
        "qwen-7b": ("Qwen/Qwen2.5-7B", 7.0, "4bit", 1, 16),
        "qwen-13b": ("Qwen/Qwen2.5-13B", 13.0, "4bit", 1, 32),
    }
    if normalized not in presets:
        raise ValueError(f"unknown preset {scale}")
    model_name, params, quantization, batch_size, grad_accum = presets[normalized]
    return build_ar_trainer_plan(
        ARModelConfig(name=model_name, parameter_count_b=params),
        LoRAConfig(quantization=quantization),
        TrainerConfig(
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=params >= 1.5,
        ),
    )


def summarize_ar_trainer_plan(run_name: str, plan: ARTrainerPlan) -> TrainReport:
    """Summarize an AR trainer plan as a smoke report."""

    return TrainReport(
        run_name=run_name,
        algorithm=AlgorithmConfig(
            name="ar_lora",
            branch="ar",
            hyperparameters={
                "rank": plan.lora.rank,
                "alpha": plan.lora.alpha,
                "quantization": plan.lora.quantization,
                "max_steps": plan.trainer.max_steps,
            },
        ),
        metrics={
            "effective_batch_size": float(plan.effective_batch_size),
            "estimated_vram_gb": plan.estimated_vram_gb,
            "learning_rate": plan.trainer.learning_rate,
        },
        artifacts={
            "model": plan.model.name,
            "tags": ",".join(plan.tags),
        },
    )
