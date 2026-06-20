# 新 idea 与 MINDRL 原论文的重复度评估

## 结论

`Interface-Native Agentic RL` 和 MINDRL 原论文有明显继承关系，但不必然重复。能否支撑一篇新论文，取决于是否把贡献从“生成模型分支接口”推进到“agentic action graph 的接口学习与控制”，并拿出原论文没有覆盖的实证对象。

我的判断：

- 如果只把 MINDRL 的 `PolicySpec -> BarrierProfile -> AdapterDecision` 换名套到 agent trajectory 上，会过于重复。
- 如果聚焦具身智能 / 世界模型 / agent action graph，定义新的 barrier probes、branch-native credit units、controller actions，并在真实或半真实 agent/VLA 环境中证明 score-routing-only 不够，则可以支撑新论文。
- 在 UniRL 已经统一多模态 RL infra 的背景下，论文重点不应是“再统一一个框架”，而应是“统一框架中缺失的接口合法性层”。

## 与原 MINDRL 的重叠

原论文已经覆盖：

- AR token branch：exact token ratio。
- parallel discrete branch：nCTC / block granularity。
- flow/diffusion branch：surrogate、anchor、drift、smoothness。
- AR+flow hybrid：离散 plan token + continuous action chunk。

因此，以下方向重复度高：

- 再提出一个通用 `branch-native controller`，但仍只在 AR/dLLM/flow 上实验。
- 只做 UniRL plugin，把原论文的 controller 接进现有训练框架。
- 只把 agent 的 tool call 当成另一个 categorical action，没有新的障碍定义或实验证据。

## 可以形成新论文的差异化

### 1. 从 generative branch 到 agent action graph

MINDRL 原论文的分支主要按生成模型结构划分。新论文可以按 agent 执行图划分：

- reasoning tokens
- tool selection
- tool arguments
- browser / environment action
- memory write
- delegation / spawn
- aggregation
- stopping

关键差异：这些分支不是单纯的模型输出模态，而是有外部状态转移、不可逆操作、延迟、成本和信息陈旧问题。

### 2. 新 barrier probes

原论文 barrier 是 nCTC、density tractability、surrogate variance、drift、smoothness。Agentic RL 可以定义新的 barrier：

- tool validity barrier：工具参数 schema、权限、执行失败。
- observation staleness barrier：网页/环境状态变化后继续用旧 observation。
- irreversibility barrier：提交、删除、支付、机器人碰撞等不可逆动作。
- evidence contradiction barrier：搜索/研究 agent 的证据互相矛盾。
- delegation redundancy barrier：多智能体重复劳动或无收益 spawn。
- stopping barrier：premature stop 与 late stop。

这些不是原论文直接处理的问题。

### 3. 新 controller actions

原论文主要调 block size、serialization、anchor、clip、branch weight、reranking。Agentic RL 可以调：

- tool retry budget
- search depth / query diversity
- browser action confirmation
- memory write threshold
- delegate/spawn budget
- aggregation consistency threshold
- stop uncertainty threshold
- world-model rollout budget

这些 action 与 agent/VLA 执行成本直接相关。

### 4. 新实证对象

原论文的强项是 language-side dLLM 和 flow/AR+flow matched evaluation。新论文应转向：

- tool-use / search / web agent
- multi-agent orchestration traces
- VLA action chunks + world model uncertainty
- embodied task stage transitions

如果能证明这些任务中 trajectory-level reward 或 score-routing-only 会诱发工具滥用、过度搜索、错误停止、世界模型 hallucination 或动作 drift，而 barrier-gated controller 能降低这些失败，就有新意。

## 更建议的最终题目

相比 `Interface-Native Agentic RL`，如果你倾向具身/世界模型，建议把题目改得更窄：

**Barrier-Gated Reinforcement Learning for World-Model-Guided Embodied Agents**

这个题目比泛 agentic RL 更不容易和 MINDRL 重复，因为它把 novelty 放在：

- world-model uncertainty / fidelity / rollout drift
- semantic-action asynchronous branches
- action chunk feasibility / contact risk
- safety envelope / irreversible action

## 推荐主线

建议主线从“泛 agentic RL”调整为“世界模型 + 具身/VLA 的接口原生 RL”：

```text
MINDRL 原论文：reward interface for generative model branches
新论文：reward interface for world-model-guided embodied control branches
```

核心贡献可以是：

1. 定义 VLA/world-model RL 中的 branch taxonomy：
   semantic reasoning、world latent、trajectory hint、action chunk、contact/safety monitor。
2. 定义 barrier probes：
   world-model uncertainty、latent drift、semantic staleness、action smoothness、contact risk。
3. 设计 barrier-gated controller：
   何时信 world model、何时刷新语义、何时缩短 action chunk、何时加强 safety envelope。
4. 在 LIBERO / CALVIN / SimplerEnv 或 world-model simulator 上验证：
   比 vanilla RL、stage reward、score routing、fixed action chunk 更稳。

## Go / No-Go

Go 条件：

- 至少两个新 barrier 与失败模式有稳定相关性。
- full controller 相比 score-routing-only 明显降低 drift/collision/stale-action/redundant-rollout。
- task success 不低于 vanilla RL，最好在 OOD 或长任务上更好。

No-Go 条件：

- barrier 只能事后解释，不能指导 controller action。
- 实验仍然主要复现原论文 AR/dLLM/flow 结果。
- 贡献变成一个工程 wrapper，而非新问题、新指标、新控制策略。
