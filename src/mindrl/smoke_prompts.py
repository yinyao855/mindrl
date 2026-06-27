"""Shared deterministic prompts for real-model smoke runs."""

from __future__ import annotations

from mindrl.peft_trainer import SFTExample


def math_smoke_examples() -> tuple[SFTExample, ...]:
    """Return tiny arithmetic prompts with strict numeric output instructions."""

    return (
        SFTExample("Question: What is 2 + 2?\nAnswer with only one number:\n", "4"),
        SFTExample("Question: What is 3 + 5?\nAnswer with only one number:\n", "8"),
    )


def math_smoke_prompts_and_answers() -> tuple[tuple[str, ...], dict[str, str]]:
    examples = math_smoke_examples()
    prompts = tuple(example.prompt for example in examples)
    answers = {example.prompt: example.target for example in examples}
    return prompts, answers
