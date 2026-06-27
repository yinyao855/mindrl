"""Run a real Hugging Face OPD smoke with privileged teacher scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindrl.hf_policy import HFCausalLMGroupPolicy, HFCausalLMTeacherSignalAdapter
from mindrl.opd import OPDConfig, run_opd_step
from mindrl.smoke_prompts import math_smoke_prompts_and_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--cache-dir", default="/gpfs/hulab/liyongqi/.cache/huggingface")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--per-token-clip", type=float, default=0.25)
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--output-dir", default="outputs/real_opd_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_model = args.teacher_model or args.student_model
    prompts, answers = math_smoke_prompts_and_answers()
    student = HFCausalLMGroupPolicy(
        model_name=args.student_model,
        device=args.device,
        local_files_only=True,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        dtype=args.dtype,
    )
    teacher = HFCausalLMTeacherSignalAdapter(
        model_name=teacher_model,
        answers=answers,
        device=args.device,
        local_files_only=True,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
    )
    result = run_opd_step(
        prompts,
        student,
        teacher,
        OPDConfig(per_token_clip=args.per_token_clip, run_name=f"real-opd-{args.student_model}"),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "real_opd_rollouts.jsonl"
    report_path = output_dir / "real_opd_report.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in result.batch.samples:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "opd_sample_loss": result.objective.sample_weights[sample.sample_id],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report_path.write_text(result.report.to_markdown(), encoding="utf-8")
    print(f"wrote {jsonl_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
