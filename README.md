# MindRL

MindRL 是一个面向生成式强化学习研究的轻量框架原型。它的核心问题不是“再造一个分布式 RL 训练系统”，而是回答：

```text
同一个 reward 应该通过什么分支原生对象，才能变成合法、稳定、可解释的 policy update？
```

AR 语言模型天然暴露 token logprob 和 policy/reference ratio，因此 PPO、GRPO、DPO 这类方法比较顺手。但 parallel discrete / dLLM、diffusion / flow、VLA 和 agentic policy 往往不暴露同一种概率对象。MindRL 把这些差异显式建模为接口障碍，并用 controller 选择合适的 adapter、granularity、anchor、clip range 和 branch weight。

英文说明见 `README_en.md`。

## 框架定位

MindRL 当前是一个 research MVP：

- 不实现 verl / OpenRLHF / TRL 那种完整分布式 runtime。
- 不负责大规模 rollout service、weight sync、actor-learner 调度。
- 专注于 reward-to-update interface layer。
- 提供可读、可测、容易接入外部训练栈的参考实现。

推荐理解为：

```text
rollout batch / rollout trace
  -> reward / teacher signal
  -> barrier profile
  -> adapter decision
  -> branch-native objective
  -> existing trainer
```

## 安装配置

项目使用 Python 3.12 和 `uv`：

```bash
cd /gpfs/hulab/liyongqi/rl/mindrl
uv sync
```

如果当前 shell 里没有 `uv`，也可以使用仓库已有虚拟环境：

```bash
/gpfs/hulab/liyongqi/rl/mindrl/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

默认依赖包括：

- `torch==2.7.1`
- `transformers>=5.12.1`
- `peft>=0.19.1`
- `accelerate>=1.14.0`

Hugging Face 模型缓存建议放在：

```text
/gpfs/hulab/liyongqi/.cache/huggingface
```

如果需要通过代理下载模型，可设置：

```bash
export http_proxy=http://10.11.0.51:7890
export https_proxy=http://10.11.0.51:7890
export HTTP_PROXY=http://10.11.0.51:7890
export HTTPS_PROXY=http://10.11.0.51:7890
```

## 快速开始

运行 dependency-light MVP smoke：

```bash
uv run python examples/run_mvp_smoke.py
uv run mindrl
```

输出写入：

```text
outputs/mvp_smoke/
```

运行完整测试：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

当前测试规模：

```text
82 tests OK
```

## 主要 Feature

### 1. 统一框架对象

`src/mindrl/core.py` 提供跨算法共用的数据结构：

- `RolloutSample` / `RolloutBatch`
- `RewardOutput`
- `TeacherSignal`
- `AlgorithmConfig`
- `ObjectiveOutput`
- `TrainReport`

这些对象让 AR、OPD、diffusion、controller ablation 的输出都能序列化成统一 report。

### 2. Interface Controller

核心接口位于 `src/mindrl/interface_controller.py`：

```text
PolicySpec -> BarrierProfile -> AdapterDecision
```

它根据分支结构和接口障碍选择 update adapter：

- AR exact-ratio update
- dependence-aware block update
- anchored flow surrogate
- score routing fallback

### 3. AR GRPO 路径

相关模块：

- `src/mindrl/grpo.py`
- `src/mindrl/ar_training.py`
- `src/mindrl/hf_policy.py`
- `examples/run_real_ar_smoke.py`

已实现：

- grouped rollout
- exact / numeric reward adapter
- group-relative advantage
- policy/reference token logprob
- sequence-level ratio
- KL diagnostic
- Markdown / JSONL report

真实模型 smoke 示例：

```bash
CUDA_VISIBLE_DEVICES=2 uv run python examples/run_real_ar_smoke.py \
  --model <local-causal-lm-snapshot> \
  --device cuda \
  --dtype fp16 \
  --group-size 2 \
  --max-new-tokens 8 \
  --output-dir outputs/real_ar_smoke
```

### 4. OPD Scoring 与 OPD Update

相关模块：

- `src/mindrl/opd.py`
- `src/mindrl/hf_policy.py`
- `src/mindrl/peft_trainer.py`
- `examples/run_real_opd_smoke.py`
- `examples/run_peft_opd_smoke.py`

已实现：

- student rollout
- privileged teacher context
- teacher token logprob scoring
- teacher entropy diagnostic
- clipped OPD objective
- PEFT LoRA 一步 OPD update
- before/after OPD loss 和 numeric reward

### 5. PEFT LoRA SFT Update

相关模块：

- `src/mindrl/peft_trainer.py`
- `examples/run_peft_sft_smoke.py`

已实现：

- LoRA target module 自动推断
- `auto` / `bf16` / `fp16` / `fp32` dtype
- before/after loss
- before/after numeric reward
- trainable parameter count

示例：

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

### 6. Diffusion / Flow / dLLM / VLA 接口原型

MindRL 还包含多类 dependency-light 原型：

- `diffusion_training.py`：DDPO-style trajectory objective、compressibility reward、prompt/caption overlap reward。
- `diffusion_adapter.py`：diffusers-style pipeline protocol 和 image-grid manifest。
- `flow_diffusion_interface.py` / `flow_surrogate.py`：flow surrogate、anchor、drift、smoothness 诊断。
- `discrete_interface.py` / `dllm_decoding.py` / `ar_proxy_nctc.py`：parallel discrete 与 nCTC proxy。
- `agentic_barriers.py`：tool / delegation / stop failure metrics。
- `vla_interface.py`：VLA branch interface 原型。

## 真实模型结果

详细结果见：

- `docs/real_results_2026_06_26_gpu2.md`
- `docs/real_results_2026_06_27_qwen_7b_14b.md`

### Qwen3-0.6B

已验证：

- AR GRPO smoke
- OPD scoring smoke
- PEFT SFT 一步更新
- PEFT OPD 一步更新

代表结果：

- PEFT SFT loss: `5.7868 -> 5.6544`
- PEFT OPD loss: `0.2207 -> 0.2139`
- OPD clipped token ratio: `0.75`

### Qwen2.5-7B-Instruct

已验证：

- fp16 单卡 load-only
- AR GRPO smoke
- PEFT SFT 一步更新

代表结果：

- load-only 显存约 `14.23GB`
- AR smoke: `reward_mean=1.0`, `kl=16.97`
- PEFT SFT loss: `4.1299 -> 4.1065`

### Qwen2.5-14B-Instruct

已验证：

- fp16 两卡 `device_map=auto` load-only
- 短 inference

代表结果：

- completion: `4\nYou are an AI assistant.`
- 显存约 `12.74GB / 14.79GB`

14B PEFT 还没有实现。当前 PEFT helper 是单 device 设计，14B 训练需要 device-map-aware PEFT、量化、CPU offload 或多卡训练路径。

## Reward 差异与 Policy Term

GRPO 的有效学习信号来自同组 rollout 的 reward 差异。

如果同一个 prompt 下两个 response 分别得到：

```text
reward(A) = 1.0
reward(B) = 0.0
```

那么 group mean 是 `0.5`，advantage 分别是：

```text
advantage(A) = 0.5
advantage(B) = -0.5
```

MindRL 当前的 GRPO policy term 近似为：

```text
policy_term = mean(logprob_ratio(sample) * advantage(sample))
```

如果所有 response 都答对，reward 全是 `1.0`，那么所有 advantage 都是 `0`，`policy_term` 也会是 `0`。这不是 ratio/KL 没工作，而是 batch 里没有偏好差异。

下一步要让真实 smoke 出现非零 `policy_term`，需要更难的 deterministic math set、更严格的 reward，或更大的 group size。

## 目录结构

```text
src/mindrl/
  core.py                    # rollout、reward、teacher signal、report 对象
  interface_controller.py    # PolicySpec -> BarrierProfile -> AdapterDecision
  ar_training.py             # GRPO、OPD、exact reward、batch baseline objective
  grpo.py                    # grouped AR rollout/reward loop
  opd.py                     # on-policy distillation loop
  hf_policy.py               # Hugging Face causal LM rollout / teacher adapter
  peft_trainer.py            # PEFT SFT / OPD LoRA update
  smoke_prompts.py           # 真实模型 smoke prompts
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

## 文档

- `docs/quickstart_ar_grpo.md`
- `docs/quickstart_ar_opd.md`
- `docs/quickstart_ar_lora.md`
- `docs/quickstart_diffusion_ddpo.md`
- `docs/custom_reward.md`
- `docs/troubleshooting.md`
- `docs/framework_integration_decision.md`
- `docs/real_results_2026_06_26_gpu2.md`
- `docs/real_results_2026_06_27_qwen_7b_14b.md`

## 当前限制与下一步

当前限制：

- tiny math prompts 对 Qwen3-0.6B / Qwen2.5-7B 太简单，reward 容易饱和。
- 14B 只有 load/inference path，暂无 PEFT path。
- LLaDA 8B 与当前 `transformers` 版本不兼容。
- GPFS 加载大模型非常慢，7B/14B 权重加载占主要时间。

建议下一步：

- 扩充 deterministic math set。
- 增加 stricter numeric reward，惩罚多余解释文本。
- 给 real AR smoke 增加 `device_map` 支持。
- 给 14B 增加量化 / offload / 多卡 PEFT path。
