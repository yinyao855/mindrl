# Qwen 7B / 14B Real Results: 2026-06-27

## Environment

- Proxy used explicitly:
  - `http_proxy=http://10.11.0.51:7890`
  - `https_proxy=http://10.11.0.51:7890`
- Hugging Face cache:
  - `/gpfs/hulab/liyongqi/.cache/huggingface`
- Main package:
  - `/gpfs/hulab/liyongqi/rl/mindrl`
- GPU state before runs:
  - GPU 2-9 were free RTX 3090 24GB cards.
- Note:
  - GPFS model loading was slow. Loading Qwen2.5-7B took about 24-26 minutes per process.
  - Loading Qwen2.5-14B took about 48 minutes.

## Downloaded Models

### Qwen2.5-7B-Instruct

Downloaded with proxy:

```bash
env http_proxy=http://10.11.0.51:7890 \
    https_proxy=http://10.11.0.51:7890 \
    HTTP_PROXY=http://10.11.0.51:7890 \
    HTTPS_PROXY=http://10.11.0.51:7890 \
    HF_HOME=/gpfs/hulab/liyongqi/.cache/huggingface \
    .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    cache_dir="/gpfs/hulab/liyongqi/.cache/huggingface/hub",
))
PY
```

Snapshot:

```text
/gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

### Qwen2.5-14B-Instruct

Downloaded with proxy:

```bash
env http_proxy=http://10.11.0.51:7890 \
    https_proxy=http://10.11.0.51:7890 \
    HTTP_PROXY=http://10.11.0.51:7890 \
    HTTPS_PROXY=http://10.11.0.51:7890 \
    HF_HOME=/gpfs/hulab/liyongqi/.cache/huggingface \
    .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    cache_dir="/gpfs/hulab/liyongqi/.cache/huggingface/hub",
))
PY
```

Snapshot:

```text
/gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
```

## Qwen2.5-7B Load-Only Preflight

Command shape:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "<Qwen2.5-7B snapshot>"
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
).to("cuda").eval()
inputs = tok("Question: What is 2 + 2?\nAnswer with only one number:\n", return_tensors="pt").to("cuda")
out = model.generate(**inputs, do_sample=False, max_new_tokens=8, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
PY
```

Result:

- Model loaded as `Qwen2ForCausalLM`.
- CUDA memory allocated after load: `14.23GB`.
- Completion began with `4`.

Interpretation:

Qwen2.5-7B fits on one RTX 3090 in fp16 for short inference. Default fp32 loading would likely be unsafe on 24GB, so the real smoke scripts now expose `--dtype`.

## Qwen2.5-7B AR GRPO Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_real_ar_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --device cuda \
  --dtype fp16 \
  --group-size 2 \
  --max-new-tokens 8 \
  --output-dir outputs/real_ar_smoke_qwen2_5_7b_numeric
```

Outputs:

- Report: `outputs/real_ar_smoke_qwen2_5_7b_numeric/real_ar_report.md`
- Rollouts: `outputs/real_ar_smoke_qwen2_5_7b_numeric/real_ar_rollouts.jsonl`

Metrics:

- `reward_mean`: `1.0`
- `kl`: `16.97458903118968`
- `policy_term`: `0.0`
- `objective`: `-0.0`

Example rollouts:

```text
2 + 2 -> "4\n\nThe question \"What is "
2 + 2 -> "4\n\nNote that the instruction explicitly requests"
3 + 5 -> "8\n\nQuestion: Solve for x in"
3 + 5 -> "8\nYou're correct! The answer"
```

Interpretation:

The full MindRL AR rollout/reward/report path works on Qwen2.5-7B. KL is non-zero, so the policy/reference logprob path is active. The policy term remains zero because all samples received reward `1.0`; the tiny math set is too easy for this model.

## Qwen2.5-7B PEFT SFT Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_peft_sft_smoke.py \
  --model /gpfs/hulab/liyongqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --device cuda \
  --dtype fp16 \
  --max-steps 1 \
  --learning-rate 1e-4 \
  --lora-rank 2 \
  --lora-alpha 4 \
  --output-dir outputs/peft_sft_smoke_qwen2_5_7b
```

Output:

- Report: `outputs/peft_sft_smoke_qwen2_5_7b/peft_sft_report.md`

Metrics:

- `before_loss`: `4.129904747009277`
- `after_loss`: `4.106450080871582`
- `loss_delta`: `-0.023454666137695312`
- `before_reward`: `1.0`
- `after_reward`: `1.0`
- `trainable_parameters`: `630784.0`

Interpretation:

Qwen2.5-7B PEFT LoRA one-step update works on a single RTX 3090 with fp16 and conservative LoRA settings (`rank=2`, `alpha=4`). The loss decreases; reward is saturated before training.

## Qwen2.5-14B Load + Inference Smoke

Command shape:

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "<Qwen2.5-14B snapshot>"
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "22GiB", 1: "22GiB", "cpu": "96GiB"},
).eval()
inputs = tok("Question: What is 2 + 2?\nAnswer with only one number:\n", return_tensors="pt")
first_device = next(model.parameters()).device
inputs = {k: v.to(first_device) for k, v in inputs.items()}
out = model.generate(**inputs, do_sample=False, max_new_tokens=8, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

Output record:

- `outputs/qwen2_5_14b_load_inference_smoke.json`

Result:

- Completion: `4\nYou are an AI assistant.`
- Device map split:
  - layers `0-21` on GPU 2
  - layers `22-47`, norm, rotary emb, lm head on GPU 3
- CUDA memory allocated inside process:
  - local GPU 0: `12.74GB`
  - local GPU 1: `14.79GB`

Interpretation:

Qwen2.5-14B does not need to be forced onto one 24GB GPU. With `device_map=auto` and two RTX 3090 cards, load-only and short inference succeed. Full PEFT update was not attempted because the current MindRL PEFT helper is single-device oriented and 14B training would need a multi-GPU/device-map-aware PEFT path or quantized/offloaded setup.

## Current Conclusion

- Proxy works for Hugging Face downloads when set explicitly through environment variables.
- Qwen2.5-7B:
  - download: success
  - load-only: success
  - AR GRPO smoke: success
  - PEFT SFT one-step update: success
- Qwen2.5-14B:
  - download: success
  - two-GPU load-only: success
  - short inference: success
  - PEFT training: not attempted yet; requires multi-GPU or memory-saving training path.

## Next Recommended Work

1. Add a harder deterministic math set so `reward_mean` does not saturate at `1.0`.
2. Add a stricter numeric reward that penalizes non-numeric continuation after the answer.
3. Add `device_map` support to inference smoke helpers for 14B.
4. Add a separate 14B PEFT path using quantization, CPU offload, or multi-GPU device map.
5. Cache model files on faster local storage when possible; GPFS loading dominates wall-clock time.
