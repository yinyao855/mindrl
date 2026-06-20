"""Small task sample utilities for low-cost MINDRL reproduction pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DependencyGroup = Literal["high", "low", "unknown"]


@dataclass(frozen=True)
class TaskSample:
    """A prompt/completion pair for teacher-forced scorer probes."""

    task: str
    prompt: str
    completion: str
    dependency_group: DependencyGroup = "unknown"


PRESET_SAMPLES: tuple[TaskSample, ...] = (
    TaskSample(
        task="gsm8k_smoke",
        dependency_group="high",
        prompt=(
            "Question: Maria has 12 apples. She gives 3 apples to each of "
            "two friends and then buys 5 more. How many apples does she have?\n"
            "Answer:"
        ),
        completion=" Maria has 12 - 6 + 5 = 11 apples.",
    ),
    TaskSample(
        task="humaneval_smoke",
        dependency_group="high",
        prompt=(
            "Complete the Python function:\n"
            "def has_close_elements(numbers, threshold):\n"
            '    """Return True if two numbers are closer than threshold."""\n'
        ),
        completion=(
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        ),
    ),
    TaskSample(
        task="hellaswag_smoke",
        dependency_group="low",
        prompt=(
            "A person opens a cookbook and places vegetables on a cutting board. "
            "The most likely next event is"
        ),
        completion=" slicing the vegetables with a kitchen knife.",
    ),
    TaskSample(
        task="lambada_smoke",
        dependency_group="low",
        prompt=(
            "The child put the book back on the shelf after reading the last "
            "page of the"
        ),
        completion=" story.",
    ),
)


def load_preset_samples(max_examples: int | None = None) -> list[TaskSample]:
    """Return built-in smoke samples."""

    return _limit_samples(list(PRESET_SAMPLES), max_examples)


def load_jsonl_samples(path: str | Path, max_examples: int | None = None) -> list[TaskSample]:
    """Load task samples from JSONL records with task, prompt, and completion."""

    samples: list[TaskSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                samples.append(
                    TaskSample(
                        task=str(record["task"]),
                        prompt=str(record["prompt"]),
                        completion=str(record["completion"]),
                        dependency_group=_parse_dependency_group(
                            record.get("dependency_group", "unknown")
                        ),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{line_number} is missing required field {exc.args[0]!r}"
                ) from exc
            if max_examples is not None and len(samples) >= max_examples:
                break
    return samples


def _parse_dependency_group(value: object) -> DependencyGroup:
    if value in {"high", "low", "unknown"}:
        return value  # type: ignore[return-value]
    raise ValueError("dependency_group must be one of: high, low, unknown")


def _limit_samples(
    samples: list[TaskSample],
    max_examples: int | None,
) -> list[TaskSample]:
    if max_examples is None:
        return samples
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    return samples[:max_examples]
