"""Run a lightweight fixed-vs-adaptive dLLM block pilot."""

from __future__ import annotations

import argparse

from mindrl_repo.benchmark_tasks import load_jsonl_samples, load_preset_samples
from mindrl_repo.dllm_decoding import (
    MockDLMDecoder,
    compare_fixed_and_adaptive,
    dllm_reproduction_command,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["preset", "jsonl"], default="preset")
    parser.add_argument("--jsonl-path")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--fixed-block-size", type=int, default=8)
    parser.add_argument("--adaptive-alpha", type=float, default=0.4)
    parser.add_argument("--print-llada-command", action="store_true")
    args = parser.parse_args()

    samples = (
        load_preset_samples(args.max_examples)
        if args.sample == "preset"
        else load_jsonl_samples(_required_jsonl(args.jsonl_path), args.max_examples)
    )
    print("task\tgroup\tfixed_score\tadaptive_score\tfixed_block\tadaptive_block")
    for sample in samples:
        uncertainty = 0.10 if sample.dependency_group == "high" else 0.02
        decoder = MockDLMDecoder(task=sample.task, uncertainty=uncertainty)
        fixed, adaptive = compare_fixed_and_adaptive(
            decoder,
            sample.prompt,
            fixed_block_size=args.fixed_block_size,
            adaptive_alpha=args.adaptive_alpha,
        )
        print(
            f"{sample.task}\t{sample.dependency_group}\t"
            f"{fixed.score:.4f}\t{adaptive.score:.4f}\t"
            f"{fixed.average_block_size:.1f}\t{adaptive.average_block_size:.1f}"
        )

    if args.print_llada_command:
        print("llada_command_template")
        print(dllm_reproduction_command())


def _required_jsonl(path: str | None) -> str:
    if path is None:
        raise ValueError("--jsonl-path is required when --sample jsonl")
    return path


if __name__ == "__main__":
    main()
