# 当前主流模型强化学习路线梳理

本文档整理 LLM、AR+diffusion / unified multimodal、具身智能 / 世界模型中的主流 RL 做法，并分析 MINDRL-style interface controller 的切入点。

## 1. LLM RL

### 主流 pipeline

LLM 后训练通常是：

```text
pretraining -> SFT -> preference / verifier reward -> policy optimization
```

主要方法：

- **PPO / RLHF**：训练 reward model，用 PPO 更新 policy，并用 KL penalty 限制偏离 reference model。
- **DPO / IPO / KTO / SimPO**：把偏好数据转成直接优化目标，不显式跑在线 RL rollout。
- **GRPO / RLVR**：对同一 prompt 采样一组 responses，用组内 reward 均值/方差估计 advantage，不需要 critic。适合数学、代码、推理这类 verifiable reward。
- **Process reward / outcome reward**：outcome reward 看最终答案，process reward 给中间推理步骤监督。
- **test-time scaling + RL**：用 best-of-N、majority voting、self-consistency 或 verifier 产生 pseudo reward，再做 test-time 或 online improvement。

### 为什么 LLM RL 好做

AR LLM 暴露 token logprob：

```text
pi(y | x) = product_t pi(y_t | x, y_<t)
```

因此 PPO/GRPO/DPO 都有清楚的 token ratio 或 sequence ratio。MINDRL 原论文指出，AR LLM 的成功容易让人误以为 reward 是通用接口，但实际上只是 AR branch 正好暴露了 exact score object。

### 当前痛点

- outcome reward 稀疏，容易 reward hacking。
- long-horizon agent 中 credit assignment 难。
- tool / browser / memory action 不只是 token，外部状态转移会带来 stale observation 和不可逆风险。
- 多轮任务中同一个 scalar reward 分到所有 token/动作，可能导致过度搜索、无效工具调用或错误停止。

## 2. AR + Diffusion / Flow / Unified Multimodal RL

### 主流模型形态

当前统一多模态模型大致有三类：

1. **纯 AR unified model**
   - 文本、图像 token、动作 token 都离散化后自回归生成。
   - 优点是 logprob 清楚，能直接做 token-level RL。
   - 缺点是高维连续输出离散化后效率和质量可能受限。

2. **AR + diffusion / flow hybrid**
   - AR 分支负责文本、reasoning、planning、prompt expansion。
   - diffusion / flow 分支负责图像、视频、动作 chunk。
   - 代表问题是两个分支的 RL 接口不同：AR 有 exact ratio，flow/diffusion 往往没有便宜 density ratio。

3. **joint LM + DM experts**
   - 语言专家和 diffusion expert 在统一模型里协同。
   - RL 可能同时优化理解、生成和相互反馈。

### 主流 RL 方法

- **DDPO / Diffusion PPO**：把 denoising steps 当作 MDP steps，用 policy gradient 优化 reward。
- **Flow-GRPO**：把 deterministic ODE flow 转成 SDE，引入探索，使 flow matching model 可做在线 GRPO。
- **FPO / flow matching policy gradients**：用 flow-native surrogate 替代 exact density ratio。
- **UniGRPO**：把 reasoning text + image generation 统一为 MDP，文本侧用 GRPO，图像侧用 FlowGRPO 或类似 flow adapter。
- **UniRL / UniRL-Zero 类框架**：把 LLM、VLM、diffusion、prompt enhancer、unified AR+diffusion 纳入统一训练 infrastructure。

### UniRL 已经覆盖什么

UniRL 的重点是统一多模态 RL 基建：

- domain entrypoints：`train_diffusion`、`train_ar`、`train_pe`、`train_unified_model`。
- rollout -> reward -> advantage -> train -> weight sync 主循环。
- 支持 diffusion、AR、prompt enhancer、unified AR+diffusion。
- 支持 GRPO、DRPO、Flow-DPPO 等算法形态。

这意味着新工作如果只是再做“统一多模态 RL 框架”，会和 UniRL 正面重叠。

### MINDRL 的切入点

MINDRL 不应该替代 UniRL，而应该作为 UniRL 缺少的接口控制层：

```text
UniRL solves: how to run multimodal RL training
MINDRL solves: which branch-native object should receive reward updates
```

更具体地说：

- AR 分支：exact token ratio。
- diffusion/flow 分支：flow-native surrogate + anchor + drift control。
- unified model：根据 branch barrier 动态调 branch weight、clip、anchor、rerank budget。

## 3. 具身智能 / VLA RL

### 主流 VLA 训练路线

典型 VLA pipeline：

```text
web-scale VLM / LLM pretraining
  -> robot trajectory / instruction-action SFT
  -> imitation learning / behavior cloning
  -> RL post-training in simulator or real environment
```

动作表达包括：

- 离散 action token。
- continuous action chunk。
- diffusion policy / flow action head。
- waypoint / trajectory hint。
- semantic stage：Reach -> Grasp -> Transport -> Place。

### 主流 RL 做法

- **Sparse binary success RL**：只用成功/失败奖励，训练 VLA 适应新任务。RIPT-VLA 是这类代表。
- **Stage-aware RL**：把长任务拆成语义阶段，每个阶段给 reward 或 preference，降低稀疏性。
- **Offline RL + BC**：从混合质量 demonstration 中学习 reward/value，稳定提取策略。
- **Online RL / interactive post-training**：在 LIBERO、CALVIN、SimplerEnv 或真实机器人中做少量交互改进。
- **Safety / intervention-aware RL**：通过人工介入、安全 envelope、失败严重度约束探索。

### 主要困难

- reward 稀疏，trajectory 成本高。
- action chunk 长时容易 drift 或错过接触时机。
- semantic reasoning 低频，控制高频，两者异步。
- world / observation 变化快，旧 semantic state 可能 stale。
- 真实机器人探索有安全风险。

## 4. 世界模型 / WAM RL

### 世界模型在 RL 中怎么用

世界模型通常提供：

- future state prediction。
- action-conditioned rollout。
- value / reward prediction。
- uncertainty / consistency signal。
- geometry / contact prior。

在 embodied/VLA 里，世界模型可以作为：

- **训练环境**：用生成的未来视频或状态模拟 rollout。
- **policy evaluator**：评估某个 action plan 的可行性。
- **planner**：在 latent space 中搜索 trajectory。
- **reward model**：提供 progress、collision、contact、goal reaching 信号。
- **safety monitor**：判断某个 action 是否越过 ODD / safety envelope。

### 主要风险

- world model hallucination：预测看似合理但物理错误。
- compounding error：多步 rollout 误差累积。
- uncertainty calibration 不准。
- policy exploit world model：在模型漏洞中拿高 reward，真实环境失败。
- 低频 semantic plan 与高频 action control 不一致。

## 5. 为什么重心应偏具身 / 世界模型

如果目标是新论文，重心更建议放在具身智能 / 世界模型，而不是泛多模态统一框架：

1. UniRL 已经把多模态 RL infrastructure 统一得比较完整。
2. MINDRL 原论文已经覆盖 AR、parallel discrete、flow/diffusion、AR+flow 的通用接口论证。
3. 具身 / 世界模型还有更具体、更难被原论文覆盖的接口障碍：
   - world latent 是否可信。
   - 何时刷新 semantic branch。
   - action chunk 多长才安全。
   - contact / collision risk 如何进入 reward interface。
   - world-model rollout budget 如何自适应分配。
4. 这些障碍有清楚的实验指标：
   - task success。
   - collision / contact failure。
   - action smoothness / jerk。
   - world prediction error。
   - OOD generalization。
   - intervention count。

## 6. 推荐论文方向

建议从原先的泛 agentic 题目收窄为：

**Barrier-Gated RL for World-Model-Guided Vision-Language-Action Agents**

核心问题：

> 当 VLA policy 同时包含 semantic branch、world latent branch 和 continuous action branch 时，共享 reward 应该如何通过不同 branch-native objects 转成可控更新？

可能贡献：

- 定义 embodied/world-model 的 branch taxonomy。
- 定义 world-model uncertainty、semantic staleness、action drift、contact risk 等 barrier probes。
- 设计 barrier-gated controller，动态调：
  - world rollout budget
  - semantic refresh rate
  - action chunk length
  - flow/diffusion anchor
  - safety envelope penalty
- 接入 UniRL 作为训练 infra，而不是重做框架。

## 参考资料入口

- UniRL: https://github.com/Tencent-Hunyuan/UniRL
- Agent Lightning: https://arxiv.org/abs/2508.03680
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- UniRL-Zero: https://arxiv.org/abs/2510.17937
- UniRL unified multimodal paper: https://arxiv.org/abs/2505.23380
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- VLA survey: https://arxiv.org/abs/2505.04769
