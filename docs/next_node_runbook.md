# Next Node Runbook

This document lists the next tasks to run after moving `mindrl` to a node with
free GPU memory. The current node can run inference smoke tests, but Qwen3-0.6B
PEFT training failed because all RTX 3090 GPUs were already heavily occupied.

## Current Repository State

- Main package: `mindrl`
- Current branch: `main`
- Relevant committed work:
  - `c429dc3` training adapters and OPD loop primitives
  - `fe546d5` real Hugging Face smoke adapters
  - `08b0232` PEFT LoRA update smoke
- Current verified tests:
  - `uv run python -m unittest discover -s tests -p "test_*.py"`
  - 77 tests passed

## Priority 0: Confirm GPU Availability

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Target:

- At least one 24GB GPU with most memory free.
- Prefer a GPU with less than 4GB used before starting Qwen3-0.6B PEFT.

## Priority 1: Reproduce Existing Inference Smokes

Run Qwen3-0.6B AR rollout:

```bash
uv run python examples/run_real_ar_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --group-size 2 \
  --max-new-tokens 16 \
  --output-dir outputs/real_ar_smoke_qwen3_0_6b_numeric
```

Expected current result:

- `reward_mean` around `0.75` on the tiny arithmetic smoke.
- This is inference/evaluation only; no model update happens here.

Run Qwen3-0.6B OPD scoring:

```bash
uv run python examples/run_real_opd_smoke.py \
  --student-model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --teacher-model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --max-new-tokens 12 \
  --per-token-clip 0.25 \
  --output-dir outputs/real_opd_smoke_qwen3_0_6b
```

Expected current result:

- OPD report includes `raw_objective`, clipped `objective`, `clipped_tokens`,
  and `mean_teacher_entropy`.
- Previous run clipped most tokens, supporting the need for token clipping before
  real updates.

## Priority 2: Run Qwen3-0.6B PEFT LoRA Update

Run:

```bash
uv run python examples/run_peft_sft_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device cuda \
  --dtype auto \
  --max-steps 1 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --output-dir outputs/peft_sft_smoke_qwen3_0_6b
```

Success criteria:

- Script completes without CUDA OOM.
- Report contains:
  - `before_loss`
  - `after_loss`
  - `loss_delta`
  - `before_reward`
  - `after_reward`
  - `trainable_parameters`
- Loss should move downward, even if reward does not improve in one step.

If still OOM:

- Try `--device cpu` only as a correctness fallback; it may be slow.
- Try smaller `--max-steps 1`, `--lora-rank 2`, `--lora-alpha 4`.
- Confirm no other process is occupying the GPU.

## Priority 3: Make GRPO Objective Non-Trivial

Current limitation:

- Real GRPO smoke uses `logprob_ratio = 1.0` and `kl = 0.0`.
- This is enough for rollout/reward evaluation but not enough for a meaningful
  policy-gradient objective.

Next task:

- Add a frozen reference model or pre-update logprob snapshot.
- Compute:
  - current policy token logprob
  - reference token logprob
  - sequence/token ratio
  - KL diagnostic

Success criteria:

- `policy_term` changes when policy/reference logprobs differ.
- Report no longer has a trivially zero GRPO objective when rewards vary.

## Priority 4: Real OPD Update

Current status:

- Student rollout and teacher scoring work.
- OPD clipping and entropy diagnostics work.
- No parameter update is performed yet.

Next task:

- Use PEFT LoRA to apply the clipped OPD token loss.
- Track:
  - raw objective
  - clipped objective
  - clipped token ratio
  - teacher entropy
  - before/after numeric reward

Success criteria:

- One-step OPD update completes on Qwen3-0.6B.
- Report shows before/after loss or OPD objective on the same mini-batch.
- Clipping ratio remains visible in the report.

## Priority 5: Improve Evaluation Prompts

Current prompts are intentionally tiny:

- `2+2=`
- `3+5=`

Next task:

- Replace with a small deterministic math set using stricter output format:

```text
Question: What is 2 + 2?
Answer with only one number:
```

Success criteria:

- Numeric reward is less sensitive to extra continuation text.
- Qwen3-0.6B baseline reward is reproducible across runs.

## Priority 6: Save Results

For each run, save:

- command used
- GPU memory state before run
- generated report markdown
- rollout JSONL when available
- short interpretation in `docs/real_results_2026_06_24.md` or a new dated doc

Suggested new file after reruns:

```text
docs/real_results_<date>_<node>.md
```

## Do Not Do Yet

- Do not start 7B/13B training until Qwen3-0.6B PEFT update is stable.
- Do not add distributed runtime code before the single-GPU path is reliable.
- Do not remove clipping/entropy diagnostics from OPD; they are central to the
  current research direction.
