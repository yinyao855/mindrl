# MINDRL 复现清单

本文档基于根目录的 `2117_Reward_Is_Not_a_Universal.pdf` 和
`reward_is_not_universal_analysis.md`，整理可优先复现的实验单元、超参和风险。

## 复现优先级

### P0：离散语言侧 nCTC + adaptive block

这是最稳的复现目标。论文中该部分主要改变固定 dLLM checkpoint 的解码策略，
不需要在线 RL 训练闭环即可验证核心 claim：

- 高 nCTC 任务中，固定大 block 的并行提交更容易退化。
- adaptive block schedule 能在 compute-neutral 条件下恢复质量。
- 低 nCTC 任务中，adaptive schedule 基本不伤害结果。

对应论文位置：

- Section 3.2：parallel discrete factorization barrier。
- Section 4：nCTC 测量与 adaptive scheduling。
- Section 6.1：GSM8K、MATH-500、HumanEval、HellaSwag、LAMBADA 主结果。
- Appendix D-J：block sampler、estimator、decode protocol、paired checkpoint、reproducibility。

### P1：controlled verification

可在 toy distribution 上复现 projection / CTC / reverse gap 的关系，以及 flow surrogate
和 anchor drift 的分离。这部分适合写成单元实验，用来验证理论直觉，但不是主系统结果。

### P2：flow / diffusion 与 AR+flow matched evaluation

完整复现成本高，依赖 Flow-Factory、EO1、LIBERO、FLUX.1-dev、reward/verifier 和
matched evaluation wrapper。建议作为后续 extension，而不是首个复现目标。

## 离散侧复现协议

论文估计量：

```text
D_hat(x; M, C) = log p_ref(x_M | C) - sum_i log p_ref(x_i | C)
nCTC_pair = D_hat / C(B, 2)
```

实现要点：

- `p_ref` 是 paired AR scorer，teacher forcing。
- block size `B = 16`。
- distance-controlled block sampling：gap `G = 4`，sampler seed `0`。
- joint term 用 chain rule progressive reveal。
- 每个 block 使用 `K = 16` 个随机 orderings，ordering seed `0`。
- 数据集使用官方 eval split。

Adaptive schedule：

```text
B_t = clip(floor(alpha / (mean_uncertainty_t + eps)), 1, B_max)
```

关键设置：

- `B_min = 1`，`B_max = 16`，`eps = 1e-5`。
- uncertainty 首选 entropy，也可比较 top-1、margin。
- `alpha` 只通过 held-out calibration set 匹配平均 forward passes，不看 test score。
- fixed baseline 为 `S = 16`，`B = 8`，token-update budget `U = 128`。
- greedy token proposal，temperature `0`。

## 算力和工程风险

paper-close 设置需要把 AR checkpoint 转成 dLLM，并在
`allenai/tulu-3-sft-mixture` 上 continuation SFT 40k steps：

- global batch size 256，sequence length 2048。
- peak LR `1e-5`，cosine decay，warmup ratio `0.03`。
- bf16，FSDP，8 x H100-80GB。

当前建议采用 minimal trend reproduction：

- 使用较小 AR scorer 或公开 dLLM checkpoint。
- 先验证 nCTC 与 block-size sensitivity 的相关性。
- 用 synthetic block distributions 复现 CTC/regret 单调性。

## 当前实现

代码位于：

- `src/mindrl_repo/discrete_interface.py`
- `src/mindrl_repo/synthetic_barriers.py`
- `tests/`

验证：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
```
