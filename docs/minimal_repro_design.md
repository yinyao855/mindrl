# Minimal nCTC + Adaptive Block 复现实验设计

目标是在没有 8 x H100 和完整 paired dLLM checkpoint 的情况下，先验证论文最核心的离散侧机制：

1. within-block dependence 可以用 CTC/nCTC 测量。
2. 高依赖 block 不适合一次性并行提交。
3. 缩小 block 或序列化会降低 factorization barrier。
4. uncertainty-guided adaptive block 是真实模型上的可执行近似。

## 当前 minimal 实现

代码：

- `src/mindrl/discrete_interface.py`
- `src/mindrl/synthetic_barriers.py`
- `examples/minimal_discrete_repro.py`
- `tests/test_discrete_interface.py`
- `tests/test_synthetic_barriers.py`

验证命令：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
```

## 从 toy 到模型的迁移

### Step 1：toy barrier

使用 finite joint distribution 直接计算 total correlation，验证：

- independent binary pair 的 CTC 接近 0。
- diagonal mass 越高，CTC 越高。
- block 从 `(0,1,2)` refine 到 `(0),(1,2)` 后 barrier 降低。
- singleton partition 的 within-block barrier 为 0。

### Step 2：teacher-forced nCTC

接入任意 AR scorer 后，把 `estimate_nctc_from_logprobs` 的输入替换为真实模型 logprob：

- marginal：`log p_ref(x_i | C)`。
- joint：`sum_k log p_ref(x_mk | C, x_m<k)`。
- 对 `K=16` 个 ordering 取平均。

这一步只需要 scorer，不需要 dLLM 解码。

### Step 3：dLLM adaptive block

在 masked/diffusion-style LM 中实现：

- fixed `B=8` baseline。
- entropy/top1/margin proxy。
- `adaptive_block_size` 决定每轮更新 block size。
- distance-controlled sampler 选择 block。

### Step 4：benchmark trend

优先选择两类任务：

- 高依赖：GSM8K、MATH-500、HumanEval。
- 低依赖：HellaSwag、LAMBADA。

目标不是第一轮完全复刻论文数值，而是复现趋势：

- high-dep 任务 nCTC 更高。
- high-dep 任务 fixed large block sensitivity 更强。
- adaptive 在 high-dep 上提升更明显。
- low-dep 上 adaptive 近似不伤害。

## 复现实验表格模板

| Task | Type | nCTC | Fixed B=8 | Adaptive | Gain | Avg fwd |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | High | TBD | TBD | TBD | TBD | TBD |
| MATH-500 | Very High | TBD | TBD | TBD | TBD | TBD |
| HumanEval | High | TBD | TBD | TBD | TBD | TBD |
| HellaSwag | Low | TBD | TBD | TBD | TBD | - |
| LAMBADA | Low | TBD | TBD | TBD | TBD | - |

## Success Criteria

minimal 阶段成功标准：

- 单元测试通过。
- toy barrier sweep 与理论方向一致。
- 至少一个真实 AR scorer 能产出 teacher-forced nCTC。
- 至少一个 masked/dLLM checkpoint 能跑 fixed vs adaptive decoding。

paper-close 阶段成功标准：

- 三个 high-dep 任务上 adaptive 平均提升显著高于 low-dep 任务。
- compute-neutral calibration 不使用 test score。
- 报告 bootstrap CI 和固定 seed。
