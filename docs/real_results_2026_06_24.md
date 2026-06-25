# Real Smoke Results: 2026-06-24

## Environment

- `torch`: 2.7.1+cu128
- `transformers`: 5.12.1
- CUDA available: yes
- Cached models used:
  - `sshleifer/tiny-gpt2`
  - `Qwen/Qwen3-0.6B`

## Real AR GRPO Smoke

Command shape:

```bash
uv run python examples/run_real_ar_smoke.py \
  --model <local snapshot> \
  --device cuda \
  --group-size 2 \
  --max-new-tokens 16
```

### tiny-gpt2

- Model generated fluent-looking but irrelevant text.
- Exact/numeric math reward: `0.0`
- `reward_mean`: `0.0`
- Interpretation: pipeline smoke passed, but model capability is too weak for math reward.

Example completions:

- `2+2=` -> `boilsMost predators WheelsGy boils mutual Singapore`
- `3+5=` -> `Bend BendMiniacious448 Pocketpublicozyg`

### Qwen3-0.6B

With numeric-prefix reward:

- 4 rollout samples
- 3 samples had the correct first numeric answer
- `reward_mean`: `0.75`

Example completions:

- `2+2=` -> `4 ...` scored `1.0`
- `3+5=` -> `8 ...` scored `1.0`

The scalar GRPO objective is still `0.0` in this smoke because no parameter
update/reference model is loaded, so the policy ratio is fixed at `1.0`. This is
expected for an evaluation-only rollout pass. The useful signal at this stage is
reward and rollout quality, not a training loss decrease.

## Real OPD Smoke

Command shape:

```bash
uv run python examples/run_real_opd_smoke.py \
  --student-model <Qwen3-0.6B snapshot> \
  --teacher-model <Qwen3-0.6B snapshot> \
  --device cuda \
  --max-new-tokens 12 \
  --per-token-clip 0.25
```

Result:

- `tokens`: `24`
- `raw_objective`: `1.9453125`
- `objective` after clipping: `0.226969401042`
- `clipped_tokens`: `20`
- `mean_teacher_entropy`: `1.785074869792`

Observation:

Most token-level teacher/student gaps were clipped. This matches the OPD concern
from the referenced blog: dense teacher signals can be dominated by high-KL
style or pivot tokens, so clipping and entropy/KL diagnostics should be treated
as first-class metrics before doing real updates.

Example OPD rollout:

- `2+2=` -> `3.6, how to calculate the sum of the numbers`
- `3+5=` -> `8, 8*1=8, 8*`

## Takeaways

- The real HF rollout path works with cached local models.
- Qwen3-0.6B produces usable math reward signal even in a tiny smoke setting.
- The current GRPO path is evaluation-only until a real optimizer/reference
  model is attached.
- The OPD path already exposes the critical diagnostics needed before training:
  raw token gap, clipped token count, and teacher entropy.

## Next Required Step

Turn the current real smoke into a real training loop:

1. Add PEFT/LoRA dependency and a minimal Qwen 0.5B/0.6B update step.
2. Add reference-model logprob or frozen pre-update policy for non-trivial GRPO
   ratios/KL.
3. Persist checkpoints and compare before/after reward on the same prompts.

## PEFT LoRA Update Smoke

After adding `peft` and `accelerate`, a real LoRA SFT update was run on cached
`sshleifer/tiny-gpt2`.

Command shape:

```bash
uv run python examples/run_peft_sft_smoke.py \
  --model <tiny-gpt2 snapshot> \
  --device cpu \
  --max-steps 5 \
  --learning-rate 1e-3
```

Result:

- `before_loss`: `10.839324951171875`
- `after_loss`: `10.839273452758789`
- `loss_delta`: `-5.14984130859375e-05`
- `trainable_parameters`: `64`
- `before_reward`: `0.0`
- `after_reward`: `0.0`

Interpretation:

- The real PEFT/LoRA update path works end to end.
- tiny-gpt2 is too weak for numeric reward improvement in this setting, but the
  training loss moves in the expected direction.

### Qwen3-0.6B Training Attempt

Qwen3-0.6B LoRA training was attempted with CUDA and automatic bf16/fp16 loading,
but failed with CUDA OOM. `nvidia-smi` showed all RTX 3090 GPUs were already
heavily occupied, with roughly 20-24GB used on each 24GB GPU.

Conclusion:

- Qwen3-0.6B inference smoke is feasible in the current environment.
- Qwen3-0.6B PEFT training needs a freer GPU, CPU fallback, or a more aggressive
  memory strategy before it can produce a meaningful before/after result.
