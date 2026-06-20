"""Run task-level AR-scorer nCTC proxy pilots.

This script produces a lightweight trend check for the MINDRL discrete-side
claim. It is an AR scorer proxy, not the full paired AR-to-dLLM protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mindrl_repo.ar_proxy_nctc import (
    NCTCProxyRecord,
    build_proxy_record,
    summarize_proxy_records,
)
from mindrl_repo.benchmark_tasks import TaskSample, load_jsonl_samples, load_preset_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--sample", choices=["preset", "jsonl"], default="preset")
    parser.add_argument("--jsonl-path")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-block-tokens", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-scorer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    samples = _load_samples(args.sample, args.jsonl_path, args.max_examples)
    if not samples:
        raise ValueError("no samples loaded")

    scorer = None
    if not args.mock_scorer:
        from mindrl_repo.hf_scorer import HFCausalLMScorer

        scorer = HFCausalLMScorer(
            model_name=args.model,
            device=args.device,
            local_files_only=args.local_files_only,
        )

    records: list[NCTCProxyRecord] = []
    for sample in samples:
        if args.mock_scorer:
            joint, marginals = _mock_score_sample(sample, args.max_block_tokens)
        else:
            assert scorer is not None
            scores = scorer.score_text_block(
                sample.prompt,
                sample.completion,
                max_block_tokens=args.max_block_tokens,
            )
            joint = scores.joint_chain_logprobs
            marginals = scores.prompt_only_marginal_logprobs
        records.append(
            build_proxy_record(
                task=sample.task,
                dependency_group=sample.dependency_group,
                joint_chain_logprobs=joint,
                prompt_only_marginal_logprobs=marginals,
            )
        )

    if args.output:
        _write_jsonl(Path(args.output), records)

    _print_summary(records)


def _load_samples(
    sample_source: str,
    jsonl_path: str | None,
    max_examples: int | None,
) -> list[TaskSample]:
    if sample_source == "preset":
        return load_preset_samples(max_examples=max_examples)
    if not jsonl_path:
        raise ValueError("--jsonl-path is required when --sample jsonl")
    return load_jsonl_samples(jsonl_path, max_examples=max_examples)


def _mock_score_sample(
    sample: TaskSample,
    max_block_tokens: int,
) -> tuple[list[float], list[float]]:
    """Deterministic scorer for CLI tests and no-model smoke runs."""

    approx_tokens = sample.completion.split()
    token_count = max(1, min(max_block_tokens, len(approx_tokens)))
    marginals = [-1.5 for _ in range(token_count)]
    dependency_bonus = 0.35 if sample.dependency_group == "high" else 0.05
    joint = [-1.5 + dependency_bonus for _ in range(token_count)]
    return joint, marginals


def _write_jsonl(path: Path, records: list[NCTCProxyRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _print_summary(records: list[NCTCProxyRecord]) -> None:
    print("task_summary")
    for summary in summarize_proxy_records(records, group_by="task"):
        print(
            f"{summary.group}\tcount={summary.count}\t"
            f"mean_pair_normalized_gap={summary.mean_pair_normalized_gap:.6f}"
        )

    print("dependency_group_summary")
    for summary in summarize_proxy_records(records, group_by="dependency_group"):
        print(
            f"{summary.group}\tcount={summary.count}\t"
            f"mean_pair_normalized_gap={summary.mean_pair_normalized_gap:.6f}"
        )


if __name__ == "__main__":
    main()
