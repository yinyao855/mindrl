"""dLLM fixed/adaptive block decoding interfaces for reproduction pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from mindrl.discrete_interface import adaptive_block_size

DecodeMode = Literal["fixed", "adaptive"]


@dataclass(frozen=True)
class DLMDecodeConfig:
    mode: DecodeMode
    block_size: int = 8
    b_min: int = 1
    b_max: int = 16
    alpha: float = 1.0
    max_steps: int = 128


@dataclass(frozen=True)
class DLMDecodeResult:
    task: str
    mode: DecodeMode
    output: str
    score: float
    steps: int
    average_block_size: float


class DLMDecoder(Protocol):
    """Minimal interface expected from LLaDA/Dream adapters."""

    def decode(self, prompt: str, config: DLMDecodeConfig) -> DLMDecodeResult:
        ...


class MockDLMDecoder:
    """Deterministic decoder used to validate fixed/adaptive evaluation logic."""

    def __init__(self, task: str, uncertainty: float, target: str = "answer") -> None:
        self.task = task
        self.uncertainty = uncertainty
        self.target = target

    def decode(self, prompt: str, config: DLMDecodeConfig) -> DLMDecodeResult:
        if config.mode == "fixed":
            block_size = config.block_size
        else:
            block_size = adaptive_block_size(
                [self.uncertainty],
                alpha=config.alpha,
                b_min=config.b_min,
                b_max=config.b_max,
            )
        penalty = self.uncertainty * max(1, block_size - 1)
        score = max(0.0, 1.0 - penalty)
        steps = min(config.max_steps, max(1, 32 // max(1, block_size)))
        return DLMDecodeResult(
            task=self.task,
            mode=config.mode,
            output=f"{prompt} {self.target}".strip(),
            score=score,
            steps=steps,
            average_block_size=float(block_size),
        )


def compare_fixed_and_adaptive(
    decoder: DLMDecoder,
    prompt: str,
    fixed_block_size: int = 8,
    adaptive_b_min: int = 1,
    adaptive_b_max: int = 16,
    adaptive_alpha: float = 1.0,
) -> tuple[DLMDecodeResult, DLMDecodeResult]:
    """Run the standard fixed-vs-adaptive comparison for one prompt."""

    fixed = decoder.decode(
        prompt,
        DLMDecodeConfig(mode="fixed", block_size=fixed_block_size),
    )
    adaptive = decoder.decode(
        prompt,
        DLMDecodeConfig(
            mode="adaptive",
            b_min=adaptive_b_min,
            b_max=adaptive_b_max,
            alpha=adaptive_alpha,
        ),
    )
    return fixed, adaptive


def dllm_reproduction_command(
    model: str = "GSAI-ML/LLaDA-8B-Instruct",
    task: str = "gsm8k",
    mode: str = "depcap",
) -> str:
    """Return a shell template for running an external LLaDA/DepCap evaluation."""

    return (
        "accelerate launch eval_llada.py "
        f"--tasks {task} "
        "--num_fewshot 5 "
        "--confirm_run_unsafe_code "
        "--model llada_dist "
        f"--model_args model_path={model},gen_length=256,"
        "L_max=128,L_min=8,lambda_u=1.2,show_speed=True "
        f"--output_path evals_results_{mode}/{task}"
    )
