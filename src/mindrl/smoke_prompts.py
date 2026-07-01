"""Shared deterministic prompts for real-model smoke runs."""

from __future__ import annotations

from mindrl.peft_trainer import SFTExample


def math_smoke_examples() -> tuple[SFTExample, ...]:
    """Return tiny arithmetic prompts with strict numeric output instructions."""

    return (
        SFTExample("Question: What is 2 + 2?\nAnswer with only one number:\n", "4"),
        SFTExample("Question: What is 3 + 5?\nAnswer with only one number:\n", "8"),
    )


def harder_math_smoke_examples() -> tuple[SFTExample, ...]:
    """Return deterministic arithmetic prompts that are less likely to saturate."""

    return (
        SFTExample("Question: What is 12 * 7?\nAnswer with only one number:\n", "84"),
        SFTExample("Question: What is 144 / 12?\nAnswer with only one number:\n", "12"),
        SFTExample("Question: What is 37 + 48?\nAnswer with only one number:\n", "85"),
        SFTExample("Question: What is 9 * 8 - 7?\nAnswer with only one number:\n", "65"),
    )


def math_smoke_prompts_and_answers(
    prompt_set: str = "basic",
) -> tuple[tuple[str, ...], dict[str, str]]:
    if prompt_set == "basic":
        examples = math_smoke_examples()
    elif prompt_set == "harder":
        examples = harder_math_smoke_examples()
    else:
        raise ValueError(f"unknown math smoke prompt_set: {prompt_set}")
    prompts = tuple(example.prompt for example in examples)
    answers = {example.prompt: example.target for example in examples}
    return prompts, answers
