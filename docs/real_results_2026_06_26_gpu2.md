# Real Smoke Results: 2026-06-26 GPU 2

## Environment

- Node had multiple free RTX 3090 GPUs.
- Runs were bound with `CUDA_VISIBLE_DEVICES=2`.
- `uv` was not on `PATH`, so commands used `.venv/bin/python`.
- Model snapshot:
  - `/gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

GPU memory before/after the main runs:

```text
0, NVIDIA GeForce RTX 3090, 21584 MiB, 24576 MiB
1, NVIDIA GeForce RTX 3090, 2494 MiB, 24576 MiB
2, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
3, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
4, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
5, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
6, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
7, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
8, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
9, NVIDIA GeForce RTX 3090, 0 MiB, 24576 MiB
```

## Prompt Update

The real-model smokes now use stricter deterministic prompts:

```text
Question: What is 2 + 2?
Answer with only one number:
```

and

```text
Question: What is 3 + 5?
Answer with only one number:
```

The shared helper is `mindrl.smoke_prompts.math_smoke_examples`.

## Real AR GRPO Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_real_ar_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --group-size 2 \
  --max-new-tokens 16 \
  --output-dir outputs/real_ar_smoke_qwen3_0_6b_numeric
```

Outputs:

- Report: `outputs/real_ar_smoke_qwen3_0_6b_numeric/real_ar_report.md`
- Rollouts: `outputs/real_ar_smoke_qwen3_0_6b_numeric/real_ar_rollouts.jsonl`

Result:

- `reward_mean`: `1.0`
- `kl`: `12.108711004257202`
- `policy_term`: `0.0`
- `objective`: `-0.0`

Interpretation:

The new reference snapshot path is active because `kl` is no longer zero. The
objective remains zero in this particular run because all four rollout samples
received reward `1.0`, so group-relative advantages sum to zero. Unit tests cover
the non-trivial case where rewards differ and policy/reference ratios change the
policy term.

## Real OPD Scoring Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_real_opd_smoke.py \
  --student-model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --teacher-model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --max-new-tokens 12 \
  --per-token-clip 0.25 \
  --output-dir outputs/real_opd_smoke_qwen3_0_6b
```

Outputs:

- Report: `outputs/real_opd_smoke_qwen3_0_6b/real_opd_report.md`
- Rollouts: `outputs/real_opd_smoke_qwen3_0_6b/real_opd_rollouts.jsonl`

Result:

- `tokens`: `24`
- `raw_objective`: `1.978002786636`
- clipped `objective`: `0.211380879084`
- `clipped_tokens`: `17`
- `mean_teacher_entropy`: `1.571411132812`

Interpretation:

Most token gaps still require clipping, confirming that clipping and teacher
entropy should stay first-class diagnostics before real OPD updates.

## PEFT SFT Update Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_peft_sft_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --dtype auto \
  --max-steps 1 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --output-dir outputs/peft_sft_smoke_qwen3_0_6b
```

Output:

- Report: `outputs/peft_sft_smoke_qwen3_0_6b/peft_sft_report.md`

Result:

- `before_loss`: `5.786800861358643`
- `after_loss`: `5.654396057128906`
- `loss_delta`: `-0.13240480422973633`
- `before_reward`: `1.0`
- `after_reward`: `1.0`
- `trainable_parameters`: `573440.0`

Interpretation:

The Qwen3-0.6B LoRA update now completes on GPU without OOM. The loss moved
downward in one step; reward was already saturated on the tiny deterministic
math set.

## PEFT OPD Update Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_peft_opd_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --teacher-model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --dtype auto \
  --max-steps 1 \
  --max-new-tokens 12 \
  --per-token-clip 0.25 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --output-dir outputs/peft_opd_smoke_qwen3_0_6b
```

Output:

- Report: `outputs/peft_opd_smoke_qwen3_0_6b/peft_opd_report.md`

Result:

- `before_loss`: `0.220703125`
- `after_loss`: `0.2138671875`
- `loss_delta`: `-0.0068359375`
- `raw_objective`: `1.93511962890625`
- `clipped_objective`: `0.220703125`
- `clipped_token_ratio`: `0.75`
- `mean_teacher_entropy`: `1.3623860677083333`
- `before_reward`: `1.0`
- `after_reward`: `1.0`
- `trainable_parameters`: `573440.0`

Interpretation:

The real OPD PEFT path completes one LoRA update on Qwen3-0.6B. The same fixed
mini-batch has lower clipped OPD loss after the update, while clipping ratio and
teacher entropy remain visible in the report.

## Verification

Focused tests:

```bash
.venv/bin/python -m unittest tests/test_smoke_prompts.py tests/test_hf_policy.py tests/test_peft_trainer.py tests/test_grpo.py tests/test_opd.py
```

Result: `18` tests passed.
