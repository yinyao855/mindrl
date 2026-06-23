# 两周 Pilot：MINDRL 复现 + 世界模型/VLA 选题验证

## 目标

两周内回答两个问题：

1. MINDRL 离散侧核心机制是否能在低成本设置下复现趋势？
2. “世界模型/VLA 的 barrier-gated reward interface” 是否比泛 agentic RL 更有新论文潜力？

## Week 1：论文复现 Pilot

### Day 1-2：toy barrier 与 nCTC 工具

已完成基础代码：

- `src/mindrl/discrete_interface.py`
- `src/mindrl/synthetic_barriers.py`
- `examples/minimal_discrete_repro.py`

验证：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
```

成功标准：

- CTC 随依赖增强而升高。
- block refinement 降低 within-block barrier。
- adaptive block size 随 uncertainty 增大而缩小。

### Day 3-4：真实 AR scorer nCTC

接入一个本地可跑 AR model，例如较小 Qwen/Llama/Gemma checkpoint：

- 从 GSM8K、HumanEval、HellaSwag 各取 50-100 个样本。
- teacher forcing 计算 marginal 和 chain-rule joint logprob。
- 输出 task-level nCTC。

成功标准：

- 高依赖任务 nCTC 明显高于低依赖任务。
- 即使 scorer 较小，排序趋势仍稳定。

### Day 5-7：masked/dLLM adaptive decoding

如果有公开 dLLM checkpoint：

- 实现 fixed `B=8`。
- 实现 entropy/top1/margin adaptive schedule。
- 小样本跑 GSM8K / HumanEval / HellaSwag。

如果没有公开 checkpoint：

- 先用 synthetic masked token simulator 替代。
- 证明 high-dep simulator 中 adaptive 优于 fixed large block。

成功标准：

- 至少产出一张 pilot 表：`Task/SimType, nCTC, Fixed, Adaptive, Gain`。
- 能判断 paper-close 复现是否值得投入算力。

## Week 2：世界模型/VLA Idea Pilot

### Day 8-9：toy agent / VLA trace benchmark

已完成基础 barrier summary：

- `src/mindrl/agentic_barriers.py`

构造三类 toy traces：

- tool-use：有效/无效工具调用、latency、utility。
- delegation：spawn、redundant delegation、late/premature stop。
- VLA/world-model：semantic staleness、action drift、contact risk、world uncertainty。

成功标准：

- 每类 trace 都能产生可解释 barrier summary。
- barrier 与任务失败模式一一对应。

### Day 10-11：controller ablation

用 `MindRLController` 比较三种策略：

- trajectory reward only。
- score-routing-only。
- full barrier-gated controller。

先不训练模型，只在 toy trace generator 上模拟 controller action：

- 限制低 utility tool retry。
- 增加高风险 action confirmation。
- 缩短高 drift action chunk。
- 提高 world uncertainty 时的 semantic refresh / rollout budget。

成功标准：

- full controller 在 cost/latency/redundancy/drift 上优于 score-routing-only。
- 不明显牺牲 task success proxy。

### Day 12-14：论文雏形判断

整理结果为四张表/图：

1. MINDRL discrete reproduction trend。
2. VLA/world-model branch taxonomy。
3. Barrier probe 与 failure mode correlation。
4. Controller ablation。

Go / No-Go 标准：

- Go：至少两个 VLA/world-model barrier probe 能稳定预测 failure，并且 full controller 比 score-routing-only 更好。
- No-Go：barrier probe 只是事后解释，无法指导 controller action。

## 两周后应产出

- `docs/mindrl_repro_checklist.md`
- `docs/minimal_repro_design.md`
- `docs/framework_integration_decision.md`
- `docs/idea_overlap_assessment.md`
- `docs/rl_landscape.md`
- `docs/two_week_pilot.md`
- 可运行 minimal code 和 tests。

## 下一步

若 pilot 成功，第三周开始：

- UniRL wrapper 或 Agent Lightning trace adapter。
- VLA/world-model toy simulator。
- LIBERO / CALVIN / SimplerEnv 小规模实验设计。
- paper draft skeleton。
