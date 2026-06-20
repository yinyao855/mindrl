"""Build a deterministic benchmark JSONL for MINDRL pilots."""

from __future__ import annotations

import argparse

from mindrl_repo.benchmark_suites import (
    load_curated_benchmark_samples,
    write_samples_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/data/benchmark_curated.jsonl")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["gsm8k", "math", "humaneval", "hellaswag", "lambada"],
    )
    parser.add_argument("--max-examples-per-task", type=int)
    args = parser.parse_args()

    samples = load_curated_benchmark_samples(
        tasks=args.tasks,
        max_examples_per_task=args.max_examples_per_task,
    )
    if not samples:
        raise ValueError("no benchmark samples selected")
    write_samples_jsonl(args.output, samples)
    print(f"wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
