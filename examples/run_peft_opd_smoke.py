"""Run a tiny real PEFT LoRA OPD update and write a report."""

from __future__ import annotations

import argparse
from pathlib import Path

from mindrl.peft_trainer import run_peft_opd_update
from mindrl.smoke_prompts import math_smoke_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--cache-dir", default="/gpfs/hulab/liyongqi/.cache/huggingface")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--per-token-clip", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--output-dir", default="outputs/peft_opd_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_peft_opd_update(
        model_name=args.model,
        teacher_model_name=args.teacher_model,
        examples=math_smoke_examples(),
        device=args.device,
        cache_dir=args.cache_dir,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        per_token_clip=args.per_token_clip,
        dtype=args.dtype,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "peft_opd_report.md"
    report_path.write_text(result.report.to_markdown(), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
