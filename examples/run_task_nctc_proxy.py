"""Run task-level AR-scorer nCTC proxy pilots.

This script produces a lightweight trend check for the MINDRL discrete-side
claim. It is an AR scorer proxy, not the full paired AR-to-dLLM protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mindrl.ar_proxy_nctc import (
    BlockNCTCRecord,
    NCTCProxyRecord,
    build_block_nctc_record,
    build_proxy_record,
    default_block_starts,
    summarize_proxy_records,
)
from mindrl.benchmark_tasks import TaskSample, load_jsonl_samples, load_preset_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--sample", choices=["preset", "jsonl"], default="preset")
    parser.add_argument("--jsonl-path")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-block-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--block-stride", type=int)
    parser.add_argument("--order-count", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-scorer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    samples = _load_samples(args.sample, args.jsonl_path, args.max_examples)
    if not samples:
        raise ValueError("no samples loaded")

    scorer = None
    if not args.mock_scorer:
        from mindrl.hf_scorer import HFCausalLMScorer

        scorer = HFCausalLMScorer(
            model_name=args.model,
            device=args.device,
            local_files_only=args.local_files_only,
        )

    records: list[NCTCProxyRecord | BlockNCTCRecord] = []
    for sample in samples:
        if args.block_size is not None:
            if args.mock_scorer:
                records.extend(
                    _mock_block_records(
                        sample,
                        block_size=args.block_size,
                        stride=args.block_stride,
                        order_count=args.order_count,
                        max_block_tokens=args.max_block_tokens,
                    )
                )
            else:
                assert scorer is not None
                records.extend(
                    _score_hf_block_records(
                        scorer,
                        sample,
                        block_size=args.block_size,
                        stride=args.block_stride,
                        order_count=args.order_count,
                    )
                )
            continue

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


def _mock_block_records(
    sample: TaskSample,
    block_size: int,
    stride: int | None,
    order_count: int,
    max_block_tokens: int,
) -> list[BlockNCTCRecord]:
    """Deterministic block records for tests and no-model smoke runs."""

    approx_tokens = sample.completion.split()
    token_count = max(1, min(max_block_tokens, len(approx_tokens)))
    starts = default_block_starts(token_count, block_size, stride)
    if not starts:
        starts = [0]
        block_size = min(block_size, token_count)

    records: list[BlockNCTCRecord] = []
    for start in starts:
        actual_size = min(block_size, token_count - start)
        marginals = [-1.5 for _ in range(actual_size)]
        dependency_bonus = 0.35 if sample.dependency_group == "high" else 0.05
        joint_orders = [
            [-1.5 + dependency_bonus for _ in range(actual_size)]
            for _ in range(order_count)
        ]
        records.append(
            build_block_nctc_record(
                task=sample.task,
                dependency_group=sample.dependency_group,
                block_start=start,
                joint_order_logprobs=joint_orders,
                marginal_logprobs=marginals,
            )
        )
    return records


def _score_hf_block_records(
    scorer: object,
    sample: TaskSample,
    block_size: int,
    stride: int | None,
    order_count: int,
) -> list[BlockNCTCRecord]:
    completion_ids = scorer.tokenizer(  # type: ignore[attr-defined]
        sample.completion,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    starts = default_block_starts(len(completion_ids), block_size, stride)
    orders = _orders_for_block(block_size, order_count)

    records: list[BlockNCTCRecord] = []
    for start in starts:
        scores = scorer.score_text_block_orders(  # type: ignore[attr-defined]
            sample.prompt,
            sample.completion,
            block_start=start,
            block_size=block_size,
            orders=orders,
        )
        records.append(
            build_block_nctc_record(
                task=sample.task,
                dependency_group=sample.dependency_group,
                block_start=scores.block_start,
                joint_order_logprobs=scores.joint_order_logprobs,
                marginal_logprobs=scores.prompt_only_marginal_logprobs,
            )
        )
    return records


def _orders_for_block(block_size: int, order_count: int) -> list[list[int]]:
    if block_size < 1:
        raise ValueError("block_size must be at least one")
    if order_count < 1:
        raise ValueError("order_count must be positive")
    base = list(range(block_size))
    orders = [base]
    if order_count > 1:
        orders.append(list(reversed(base)))
    while len(orders) < order_count:
        shift = len(orders) % block_size
        orders.append(base[shift:] + base[:shift])
    return orders[:order_count]


def _write_jsonl(path: Path, records: list[NCTCProxyRecord | BlockNCTCRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _print_summary(records: list[NCTCProxyRecord | BlockNCTCRecord]) -> None:
    if records and isinstance(records[0], BlockNCTCRecord):
        print("task_summary")
        for task in sorted({record.task for record in records}):
            group_records = [record for record in records if record.task == task]
            mean_gap = sum(record.pair_normalized_nctc for record in group_records) / len(
                group_records
            )
            print(
                f"{task}\tcount={len(group_records)}\t"
                f"mean_pair_normalized_nctc={mean_gap:.6f}"
            )

        print("dependency_group_summary")
        for group in sorted({record.dependency_group for record in records}):
            group_records = [
                record for record in records if record.dependency_group == group
            ]
            mean_gap = sum(record.pair_normalized_nctc for record in group_records) / len(
                group_records
            )
            print(
                f"{group}\tcount={len(group_records)}\t"
                f"mean_pair_normalized_nctc={mean_gap:.6f}"
            )
        return

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
