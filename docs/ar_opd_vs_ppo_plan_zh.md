# AR LLM 上 OPD vs PPO-style 对照计划

## 目标

在 AR LLM 上建立一个轻量、单机可复现的对照实验，用于回答：

```text
OPD 相比 PPO/PPO-style RL 是否在同等预算下更稳定、更省样本，或者更少依赖稀疏 reward？
```

第一版不接入完整分布式 PPO，而是提供：

- OPD LoRA 更新脚本。
- PPO-style objective smoke 脚本。
- 统一 prompt set / reward mode。
- 可保存的 report 与 rollout JSONL。

## 实验对象

优先模型：

- `Qwen/Qwen3-0.6B`：迭代快，适合调参。
- `Qwen/Qwen2.5-7B-Instruct`：已验证可单卡 fp16 跑 AR smoke 和 PEFT SFT。

暂不优先：

- `Qwen2.5-14B`：已跑通两卡 inference，但 PEFT 需要 device-map-aware 训练。
- LLaDA / Dream：还存在 remote code / snapshot 问题。

## 对照方法

### OPD

使用 student 自己 rollout 的状态，teacher 在这些状态上提供 dense token logprob 指导。

核心指标：

```text
before_loss
after_loss
raw_objective
clipped_objective
clipped_token_ratio
mean_teacher_entropy
before_reward
after_reward
```

### PPO-style

第一版实现 PPO-style diagnostic objective：

```text
ratio = pi_theta(sample) / pi_ref(sample)
advantage = reward - group_mean_reward
policy_term = mean(min(ratio * advantage, clip(ratio) * advantage))
objective = -(policy_term - kl_weight * kl)
```

它不是完整 PPO trainer，但可以作为：

- reward 差异是否存在的检查。
- `policy_term` 是否非零的检查。
- 后续 PEFT PPO update 的 objective 基础。

## Prompt 与 Reward

使用：

```bash
--prompt-set harder
--reward-mode strict_numeric
```

这样更容易避免 reward 饱和，并让 `policy_term` 变成非零。

## 运行建议

先跑 Qwen3-0.6B：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_real_ar_smoke.py \
  --model <Qwen3-0.6B snapshot> \
  --device cuda \
  --dtype fp16 \
  --group-size 4 \
  --max-new-tokens 8 \
  --prompt-set harder \
  --reward-mode strict_numeric \
  --output-dir outputs/real_ar_smoke_qwen3_0_6b_harder_strict
```

然后跑 PPO-style diagnostic：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_ar_ppo_style_smoke.py \
  --model <Qwen3-0.6B snapshot> \
  --device cuda \
  --dtype fp16 \
  --group-size 4 \
  --max-new-tokens 8 \
  --prompt-set harder \
  --reward-mode strict_numeric \
  --clip-range 0.2 \
  --kl-weight 0.01 \
  --output-dir outputs/ar_ppo_style_qwen3_0_6b_harder_strict
```

最后跑 OPD update：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/run_peft_opd_smoke.py \
  --model <Qwen3-0.6B snapshot> \
  --teacher-model <Qwen3-0.6B snapshot> \
  --device cuda \
  --dtype fp16 \
  --max-steps 1 \
  --max-new-tokens 12 \
  --per-token-clip 0.25 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --output-dir outputs/peft_opd_qwen3_0_6b_harder_strict
```

## 下一步

第一版脚本跑通后，再实现真正的 PPO-style PEFT update，并和 OPD 的 before/after reward、loss、entropy、drift 做公平对比。
