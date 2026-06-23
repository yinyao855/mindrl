# MINDRL 框架定位与集成决策

## 决策

MINDRL 不应首先做成一个完整独立强化学习库，而应做成可插拔模块：

```text
rollout trace / rollout track
  -> branch decomposition
  -> barrier probes
  -> adapter decision
  -> branch-native objective terms
  -> existing train stack
```

原因：

- UniRL 已经覆盖多模态 RL post-training 主循环：rollout、reward、advantage、train、weight sync。
- Agent Lightning 已经强调 agent execution 与 RL training 解耦。
- verl / TRL 等已有成熟 LLM RL 训练栈。
- MINDRL 的核心贡献是 reward-to-update interface controller，而不是分布式训练 runtime。

## 与 UniRL 的关系

UniRL 的公开文档显示其数据流为：

```text
DomainTrainer
  -> RolloutReq
  -> RolloutResp / RolloutTrack
  -> RewardService.score_and_attach
  -> RolloutTrack.compute_advantages
  -> TrainStack.train_track
```

MINDRL 最合适插在 reward/advantage 与 train stack 之间，或者作为 algorithm 内部的前置 controller：

```text
RolloutTrack + rewards + advantages
  -> MindRLController
  -> branch-specific score/surrogate/anchor/structure terms
  -> StageAlgorithm loss
```

## 模块 API

当前轻量实现位于：

- `src/mindrl/interface_controller.py`

核心类型：

- `PolicySpec`：分支静态信息，例如 modality、structure、score availability、default granularity。
- `BarrierProfile`：实时障碍，例如 nCTC、density cost、surrogate variance、drift、smoothness。
- `AdapterDecision`：controller 输出，例如 adapter、granularity、anchor strength、structure cost weight、branch weight、clip range。
- `MindRLController`：规则型参考实现。

## UniRL Plugin 形态

建议未来提供三个 adapter，而不是 fork UniRL：

1. `UniRLRolloutBranchExtractor`
   - 输入：`RolloutTrack`。
   - 输出：若干 `PolicySpec` 与 branch payload。

2. `UniRLBarrierProbe`
   - AR：exact token ratio availability。
   - parallel discrete：nCTC 或 uncertainty proxy。
   - diffusion/flow：surrogate variance、drift、smoothness、density path availability。

3. `UniRLAlgorithmWrapper`
   - 包装现有 `StageAlgorithm`。
   - 在 loss 计算前调用 `MindRLController.decide`。
   - 把 branch weight、clip range、anchor weight、structure penalty 注入原算法。

## Agent Lightning / verl 集成形态

Agent Lightning 更适合作为 agent trajectory 来源：

- reasoning token。
- tool call。
- tool arguments。
- environment/browser action。
- memory write。
- delegation / spawn。
- aggregation。
- stop。

MINDRL 在这里不替换 hierarchical credit assignment，而是给每类 action 定义 branch-native credit unit 和 barrier budget。

verl / TRL 则更适合作为 AR exact-ratio baseline。MINDRL 可以先作为 callback 或 batch preprocessor：

- 读取 rollout metadata。
- 标注分支类型。
- 修改 per-sample/per-token loss weights。
- 加入 branch-level penalties。

## 独立库 vs 插件模块

| 方案 | 优点 | 缺点 | 判断 |
| --- | --- | --- | --- |
| 独立 RL 库 | 完全控制 API，可做端到端 demo | 重复 UniRL/verl/TRL 的 runtime，维护成本高 | 不建议作为第一步 |
| 插件模块 | 能复用现有训练栈，论文贡献更集中 | 需要适配多个框架接口 | 推荐 |
| 只做论文代码 | 最快验证 idea | 工程影响力不足 | 可作为 pilot，但不是最终形态 |

最终建议：先做 `mindrl-interface` 风格的小包，包含 controller/probe/adapter 抽象和 reference implementations；再提供 UniRL plugin 和 agentic / VLA plugin。
