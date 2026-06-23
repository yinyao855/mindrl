"""Curated benchmark samples for MINDRL reproduction pilots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from mindrl.benchmark_tasks import TaskSample


BENCHMARK_SAMPLES: tuple[TaskSample, ...] = (
    TaskSample(
        task="gsm8k",
        dependency_group="high",
        prompt=(
            "Question: A bakery made 36 muffins. It sold 14 in the morning, "
            "sold 9 more after lunch, and baked 18 more in the evening. "
            "How many muffins are left?\nAnswer:"
        ),
        completion=" The bakery has 36 - 14 - 9 + 18 = 31 muffins left.",
    ),
    TaskSample(
        task="gsm8k",
        dependency_group="high",
        prompt=(
            "Question: Noah has 5 boxes with 8 pencils in each box. He gives "
            "7 pencils away. How many pencils does he have now?\nAnswer:"
        ),
        completion=" Noah starts with 5 * 8 = 40 pencils, then has 40 - 7 = 33 pencils.",
    ),
    TaskSample(
        task="math",
        dependency_group="high",
        prompt="Problem: Solve for x: 3x + 7 = 28.\nSolution:",
        completion=" Subtract 7 from both sides to get 3x = 21, so x = 7.",
    ),
    TaskSample(
        task="math",
        dependency_group="high",
        prompt="Problem: If a rectangle has area 72 and width 8, what is its length?\nSolution:",
        completion=" The length is area divided by width, so 72 / 8 = 9.",
    ),
    TaskSample(
        task="humaneval",
        dependency_group="high",
        prompt=(
            "Complete the Python function:\n"
            "def count_vowels(text):\n"
            "    \"\"\"Return the number of vowels in text.\"\"\"\n"
        ),
        completion=(
            "    total = 0\n"
            "    for char in text.lower():\n"
            "        if char in \"aeiou\":\n"
            "            total += 1\n"
            "    return total\n"
        ),
    ),
    TaskSample(
        task="humaneval",
        dependency_group="high",
        prompt=(
            "Complete the Python function:\n"
            "def all_even(values):\n"
            "    \"\"\"Return True if all integers are even.\"\"\"\n"
        ),
        completion=(
            "    for value in values:\n"
            "        if value % 2 != 0:\n"
            "            return False\n"
            "    return True\n"
        ),
    ),
    TaskSample(
        task="hellaswag",
        dependency_group="low",
        prompt="A chef rinses tomatoes and places them on a cutting board. The chef then",
        completion=" slices the tomatoes with a sharp knife.",
    ),
    TaskSample(
        task="hellaswag",
        dependency_group="low",
        prompt="A cyclist puts on a helmet and walks to the bicycle. The cyclist then",
        completion=" gets on the bicycle and starts riding.",
    ),
    TaskSample(
        task="lambada",
        dependency_group="low",
        prompt="The musician tuned the strings before walking onto the",
        completion=" stage.",
    ),
    TaskSample(
        task="lambada",
        dependency_group="low",
        prompt="After the rain stopped, the children went outside to play in the",
        completion=" yard.",
    ),
)


def load_curated_benchmark_samples(
    tasks: Iterable[str] | None = None,
    max_examples_per_task: int | None = None,
) -> list[TaskSample]:
    """Load deterministic benchmark-like samples without network access."""

    task_filter = set(tasks) if tasks is not None else None
    counts: dict[str, int] = {}
    samples: list[TaskSample] = []
    for sample in BENCHMARK_SAMPLES:
        if task_filter is not None and sample.task not in task_filter:
            continue
        count = counts.get(sample.task, 0)
        if max_examples_per_task is not None and count >= max_examples_per_task:
            continue
        samples.append(sample)
        counts[sample.task] = count + 1
    return samples


def write_samples_jsonl(path: str | Path, samples: Iterable[TaskSample]) -> None:
    """Write task samples in the JSONL format consumed by the pilot CLI."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "task": sample.task,
                        "prompt": sample.prompt,
                        "completion": sample.completion,
                        "dependency_group": sample.dependency_group,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
