# MindRL

MindRL is a lightweight research framework for generative reinforcement learning. Its core question is not how to rebuild another distributed RL runtime, but:

```text
Through which branch-native object should a shared reward become a valid and stable policy update?
```

Autoregressive language models expose token logprobs and policy/reference ratios, so PPO, GRPO, and DPO fit naturally. Parallel discrete models, diffusion/flow policies, VLA policies, and agentic policies do not expose the same probability object. MindRL makes these interface differences explicit and uses a controller to select adapters, granularity, anchors, clip ranges, and branch weights.

Chinese documentation is in `README.md`.

## Project Positioning

MindRL is currently a research MVP:

- It does not implement a full distributed runtime like verl, OpenRLHF, TRL, or UniRL.
- It does not own rollout services, weight synchronization, or actor-learner scheduling.
- It focuses on the reward-to-update interface layer.
- It provides readable, testable reference implementations that can later plug into existing training stacks.

The intended flow is:

```text
rollout batch / rollout trace
  -> reward / teacher signal
  -> barrier profile
  -> adapter decision
  -> branch-native objective
  -> existing trainer
```

## Installation

The project uses Python 3.12 and `uv`:

```bash
cd /gpfs/hulab/liyongqi/rl/mindrl
uv sync
```

If `uv` is not available in the current shell, use the existing virtual environment:

```bash
/gpfs/hulab/liyongqi/rl/mindrl/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

Main dependencies:

- `torch==2.7.1`
- `transformers>=5.12.1`
- `peft>=0.19.1`
- `accelerate>=1.14.0`

Recommended Hugging Face cache location:

```text
/gpfs/hulab/liyongqi/.cache/huggingface
```

If model downloads need a proxy:

```bash
export http_proxy=http://10.11.0.51:7890
export https_proxy=http://10.11.0.51:7890
export HTTP_PROXY=http://10.11.0.51:7890
export HTTPS_PROXY=http://10.11.0.51:7890
```

## Quick Start

Run the dependency-light MVP smoke:

```bash
uv run python examples/run_mvp_smoke.py
uv run mindrl
```

Outputs are written to:

```text
outputs/mvp_smoke/
```

Run the full test suite:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Current test status:

```text
82 tests OK
```

## Main Features

### 1. Shared Framework Objects

`src/mindrl/core.py` defines common data structures:

- `RolloutSample` / `RolloutBatch`
- `RewardOutput`
- `TeacherSignal`
- `AlgorithmConfig`
- `ObjectiveOutput`
- `TrainReport`

These objects make reports from AR, OPD, diffusion, and controller ablation paths serializable in a common format.

### 2. Interface Controller

The core controller API lives in `src/mindrl/interface_controller.py`:

```text
PolicySpec -> BarrierProfile -> AdapterDecision
```

It selects update adapters from branch structure and interface barriers:

- AR exact-ratio update
- dependence-aware block update
- anchored flow surrogate
- score-routing fallback

### 3. AR GRPO Path

Relevant modules:

- `src/mindrl/grpo.py`
- `src/mindrl/ar_training.py`
- `src/mindrl/hf_policy.py`
- `examples/run_real_ar_smoke.py`

Implemented:

- grouped rollout
- exact / numeric reward adapters
- group-relative advantages
- policy/reference token logprobs
- sequence-level ratios
- KL diagnostics
- Markdown / JSONL reports

Real-model smoke example:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python examples/run_real_ar_smoke.py \
  --model <local-causal-lm-snapshot> \
  --device cuda \
  --dtype fp16 \
  --group-size 2 \
  --max-new-tokens 8 \
  --output-dir outputs/real_ar_smoke
```

### 4. OPD Scoring and OPD Update

Relevant modules:

- `src/mindrl/opd.py`
- `src/mindrl/hf_policy.py`
- `src/mindrl/peft_trainer.py`
- `examples/run_real_opd_smoke.py`
- `examples/run_peft_opd_smoke.py`

Implemented:

- student rollout
- privileged teacher context
- teacher token logprob scoring
- teacher entropy diagnostics
- clipped OPD objective
- one-step PEFT LoRA OPD update
- before/after OPD loss and numeric reward

### 5. PEFT LoRA SFT Update

Relevant modules:

- `src/mindrl/peft_trainer.py`
- `examples/run_peft_sft_smoke.py`

Implemented:

- automatic LoRA target module inference
- `auto` / `bf16` / `fp16` / `fp32` dtype modes
- before/after loss
- before/after numeric reward
- trainable parameter count

Example:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python examples/run_peft_sft_smoke.py \
  --model <local-causal-lm-snapshot> \
  --device cuda \
  --dtype fp16 \
  --max-steps 1 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --output-dir outputs/peft_sft_smoke
```

### 6. Diffusion / Flow / dLLM / VLA Interface Prototypes

MindRL also includes dependency-light prototypes:

- `diffusion_training.py`: DDPO-style trajectory objective, compressibility reward, prompt/caption overlap reward.
- `diffusion_adapter.py`: diffusers-style pipeline protocol and image-grid manifests.
- `flow_diffusion_interface.py` / `flow_surrogate.py`: flow surrogate, anchor, drift, smoothness diagnostics.
- `discrete_interface.py` / `dllm_decoding.py` / `ar_proxy_nctc.py`: parallel discrete and nCTC proxy.
- `agentic_barriers.py`: tool / delegation / stop failure metrics.
- `vla_interface.py`: VLA branch interface prototype.

## Real-Model Results

Detailed results:

- `docs/real_results_2026_06_26_gpu2.md`
- `docs/real_results_2026_06_27_qwen_7b_14b.md`

### Qwen3-0.6B

Verified:

- AR GRPO smoke
- OPD scoring smoke
- PEFT SFT one-step update
- PEFT OPD one-step update

Representative results:

- PEFT SFT loss: `5.7868 -> 5.6544`
- PEFT OPD loss: `0.2207 -> 0.2139`
- OPD clipped token ratio: `0.75`

### Qwen2.5-7B-Instruct

Verified:

- fp16 single-GPU load-only
- AR GRPO smoke
- PEFT SFT one-step update

Representative results:

- load-only memory: about `14.23GB`
- AR smoke: `reward_mean=1.0`, `kl=16.97`
- PEFT SFT loss: `4.1299 -> 4.1065`

### Qwen2.5-14B-Instruct

Verified:

- fp16 two-GPU `device_map=auto` load-only
- short inference

Representative results:

- completion: `4\nYou are an AI assistant.`
- memory: about `12.74GB / 14.79GB`

14B PEFT is not implemented yet. The current PEFT helper is single-device oriented; 14B training needs device-map-aware PEFT, quantization, CPU offload, or a multi-GPU training path.

## Reward Differences and Policy Term

GRPO gets useful learning signal from reward differences within a rollout group.

For one prompt, suppose two responses get:

```text
reward(A) = 1.0
reward(B) = 0.0
```

The group mean is `0.5`, so:

```text
advantage(A) = 0.5
advantage(B) = -0.5
```

MindRL's current GRPO policy term is approximately:

```text
policy_term = mean(logprob_ratio(sample) * advantage(sample))
```

If every response is correct and every reward is `1.0`, every advantage becomes `0`, so `policy_term` is also `0`. This does not mean ratio/KL is broken; it means the batch has no preference signal.

To get a non-zero policy term in real smokes, the next step is to use a harder deterministic math set, stricter reward, or larger group size.

## Repository Layout

```text
src/mindrl/
  core.py                    # rollout, reward, teacher signal, report objects
  interface_controller.py    # PolicySpec -> BarrierProfile -> AdapterDecision
  ar_training.py             # GRPO, OPD, exact reward, batch baseline objective
  grpo.py                    # grouped AR rollout/reward loop
  opd.py                     # on-policy distillation loop
  hf_policy.py               # Hugging Face causal LM rollout / teacher adapter
  peft_trainer.py            # PEFT SFT / OPD LoRA update
  smoke_prompts.py           # real-model smoke prompts
  diffusion_training.py      # DDPO-style diffusion objective
  diffusion_adapter.py       # diffusion rollout adapter
  flow_diffusion_interface.py
  discrete_interface.py
  agentic_barriers.py
examples/
  run_mvp_smoke.py
  run_real_ar_smoke.py
  run_real_opd_smoke.py
  run_peft_sft_smoke.py
  run_peft_opd_smoke.py
tests/
docs/
```

## Documentation

- `docs/quickstart_ar_grpo.md`
- `docs/quickstart_ar_opd.md`
- `docs/quickstart_ar_lora.md`
- `docs/quickstart_diffusion_ddpo.md`
- `docs/custom_reward.md`
- `docs/troubleshooting.md`
- `docs/framework_integration_decision.md`
- `docs/real_results_2026_06_26_gpu2.md`
- `docs/real_results_2026_06_27_qwen_7b_14b.md`

## Current Limitations and Next Steps

Limitations:

- The tiny math prompts are too easy for Qwen3-0.6B and Qwen2.5-7B, so reward saturates quickly.
- 14B has a load/inference path, but no PEFT path yet.
- LLaDA 8B is incompatible with the current `transformers` version.
- GPFS model loading dominates wall-clock time for large models.

Recommended next steps:

- Expand the deterministic math set.
- Add a stricter numeric reward that penalizes verbose continuations.
- Add `device_map` support to real AR smoke helpers.
- Add quantized / offloaded / multi-GPU PEFT for 14B.
