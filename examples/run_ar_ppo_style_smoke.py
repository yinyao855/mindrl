"""Run a real Hugging Face AR rollout through a PPO-style diagnostic objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindrl.grpo import NumericAnswerRewardAdapter, StrictNumericAnswerRewardAdapter
from mindrl.hf_policy import HFCausalLMGroupPolicy
from mindrl.ppo_style import compute_ppo_style_objective
from mindrl.smoke_prompts import math_smoke_prompts_and_answers
from mindrl.core import AlgorithmConfig, TrainReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--cache-dir", default="/gpfs/hulab/liyongqi/.cache/huggingface")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--prompt-set", default="basic", choices=("basic", "harder"))
    parser.add_argument("--reward-mode", default="numeric", choices=("numeric", "strict_numeric"))
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--output-dir", default="outputs/ar_ppo_style_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts, answers = math_smoke_prompts_and_answers(args.prompt_set)
    policy = HFCausalLMGroupPolicy(
        model_name=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        dtype=args.dtype,
    )
    batch = policy.rollout(prompts, args.group_size)
    rewards = _reward_adapter(args.reward_mode, answers).score(batch)
    objective = compute_ppo_style_objective(
        batch=batch,
        rewards=rewards,
        logprob_ratios=policy.logprob_ratios(batch),
        kl_by_sample=policy.kl(batch),
        clip_range=args.clip_range,
        kl_weight=args.kl_weight,
    )
    report = TrainReport(
        run_name=f"ar-ppo-style-{args.model}",
        algorithm=AlgorithmConfig(
            name="ppo_style",
            branch="ar",
            hyperparameters={
                "group_size": args.group_size,
                "clip_range": args.clip_range,
                "kl_weight": args.kl_weight,
                "prompt_set": args.prompt_set,
                "reward_mode": args.reward_mode,
            },
        ),
        metrics={
            "objective": objective.objective,
            "reward_mean": objective.diagnostics["reward_mean"],
            "policy_term": objective.diagnostics["policy_term"],
            "clipped_policy_term": objective.diagnostics["clipped_policy_term"],
            "kl": objective.diagnostics["kl"],
        },
    )

    sequence_logprobs = policy.sequence_logprobs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "ppo_style_rollouts.jsonl"
    report_path = output_dir / "ppo_style_report.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in batch.samples:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "reward": rewards.sample_rewards[sample.sample_id],
                        "advantage": objective.sample_weights[sample.sample_id],
                        "sequence_logprob": sequence_logprobs.get(sample.sample_id, 0.0),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"wrote {jsonl_path}")
    print(f"wrote {report_path}")


def _reward_adapter(mode: str, answers: dict[str, str]):
    if mode == "numeric":
        return NumericAnswerRewardAdapter(answers)
    if mode == "strict_numeric":
        return StrictNumericAnswerRewardAdapter(answers)
    raise ValueError(f"unknown reward mode: {mode}")


if __name__ == "__main__":
    main()
