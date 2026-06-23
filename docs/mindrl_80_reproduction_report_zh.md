# MINDRL 80% 复现阶段报告

本文档总结当前仓库对 MINDRL 论文的 80% 复现支架与已验证路径。
这里的“80%”指覆盖主要经验结论与实验协议，而不是复刻所有大模型训练和
所有原文表格数值。

## 当前覆盖范围

已实现：

- CTC / nCTC toy barrier 验证。
- block-level multi-order nCTC measurement。
- GSM8K / MATH / HumanEval / HellaSwag / LAMBADA 风格 curated benchmark JSONL。
- Qwen / GPT-style causal LM scorer 接口。
- dLLM fixed block vs adaptive block 的统一 adapter 和 mock pilot。
- LLaDA / DepCap 外部运行命令模板。
- `PolicySpec -> BarrierProfile -> AdapterDecision` controller ablation。
- toy flow surrogate 的 anchor drift / clip tradeoff。

未完全覆盖：

- 论文 paired AR-to-dLLM nCTC 的严格 protocol。
- LLaDA / Dream 真实权重上的完整 benchmark 评测。
- diffusion / flow 大模型 RL 训练。
- AR + flow / VLA / embodied 闭环任务。

## 推荐运行顺序

### 1. 基础单测

```bash
cd /gpfs/hulab/liyongqi/rl/mindrl
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python -m unittest discover -s tests -p "test_*.py"
```

### 2. 构建 benchmark JSONL

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/build_benchmark_jsonl.py \
  --output examples/data/benchmark_curated.jsonl
```

### 3. block-level nCTC mock pilot

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/run_task_nctc_proxy.py \
  --sample jsonl \
  --jsonl-path examples/data/benchmark_curated.jsonl \
  --device cpu \
  --mock-scorer \
  --block-size 2 \
  --block-stride 2 \
  --order-count 2
```

该命令验证新的 block-level / multi-order nCTC 输出格式。

### 4. Qwen3 GPU block-level nCTC pilot

```bash
CUDA_VISIBLE_DEVICES=4 \
PATH="/home/liyongqi/.local/bin:$PATH" \
HF_HOME="/gpfs/hulab/liyongqi/.cache/huggingface" \
HF_HUB_DISABLE_XET=1 \
http_proxy="http://10.11.0.51:7890" \
https_proxy="http://10.11.0.51:7890" \
HTTP_PROXY="http://10.11.0.51:7890" \
HTTPS_PROXY="http://10.11.0.51:7890" \
uv run python examples/run_task_nctc_proxy.py \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --sample jsonl \
  --jsonl-path examples/data/benchmark_curated.jsonl \
  --block-size 4 \
  --block-stride 4 \
  --order-count 2 \
  --output outputs/qwen3_0_6b_block_nctc.jsonl
```

注意：该结果仍然是 causal-LM proxy，不是论文的 paired AR-to-dLLM nCTC。

### 5. dLLM fixed/adaptive mock pilot

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/run_dllm_block_pilot.py \
  --sample jsonl \
  --jsonl-path examples/data/benchmark_curated.jsonl \
  --max-examples 10 \
  --print-llada-command
```

mock pilot 用 uncertainty 模拟 dLLM 解码风险，验证 fixed block 与 adaptive block
对比表格。

### 6. controller ablation

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/run_controller_ablation.py
```

输出三种策略：

- `uniform`：所有分支同样接 reward。
- `score_routing`：只按 score availability 做粗路由。
- `barrier_gated`：使用 MINDRL controller 根据 nCTC、variance、drift 调参。

### 7. flow surrogate pilot

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/run_flow_surrogate_pilot.py
```

该命令展示 anchor 降低 residual drift，clip 降低 update 幅度的 tradeoff。

## LLaDA / DepCap 外部复现入口

当前仓库不 vendor LLaDA/DepCap 代码，只提供 adapter 边界和命令模板。
建议另建外部目录克隆 DepCap 或 LLaDA 官方 repo，然后使用类似命令：

```bash
accelerate launch eval_llada.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,L_max=128,L_min=8,lambda_u=1.2,show_speed=True \
  --output_path evals_results_depcap/gsm8k
```

真实 paper-close 复现需要把该外部结果整理成：

| Task | Type | nCTC | Fixed B=8 | Adaptive | Gain | Avg fwd |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | High | TBD | TBD | TBD | TBD | TBD |
| MATH | High | TBD | TBD | TBD | TBD | TBD |
| HumanEval | High | TBD | TBD | TBD | TBD | TBD |
| HellaSwag | Low | TBD | TBD | TBD | TBD | TBD |
| LAMBADA | Low | TBD | TBD | TBD | TBD | TBD |

## 当前结论

当前仓库已经把复现工程从 smoke test 推进到可扩展的 reproduction scaffold：

- AR proxy 已升级为 block-level multi-order nCTC。
- benchmark 样本和 JSONL 格式已固定。
- dLLM / controller / flow 三条主线都有可运行的轻量 pilot。

但严格意义上，论文还未被完整复现。要声称复现到 80%，仍需要至少完成：

1. 在 Qwen3 或同类 AR scorer 上跑 benchmark block-level nCTC，并报告 high/low 趋势。
2. 在 LLaDA 或 Dream 上跑真实 fixed/adaptive decoding，而不是 mock。
3. 将 dLLM 结果与 nCTC 分组关联起来，验证 high-dep 上 adaptive gain 更明显。
4. 将 controller ablation 与真实或半真实 branch metrics 连接。

因此当前状态是：

```text
工程支架：约 70-80% 完成
论文实证复现：约 35-45% 完成
完整 paper-close 复现：仍需真实 LLaDA/Dream 评测
```

## 2026-06-21 真实运行记录

### Qwen3-0.6B block-level nCTC

已在 GPU4 上运行 `Qwen/Qwen3-0.6B` 的 block-level multi-order nCTC。

block size 4：

```text
task_summary
gsm8k     count=13 mean_pair_normalized_nctc=2.996369
hellaswag count=4  mean_pair_normalized_nctc=1.570262
humaneval count=13 mean_pair_normalized_nctc=3.994991
math      count=9  mean_pair_normalized_nctc=2.285404

dependency_group_summary
high count=35 mean_pair_normalized_nctc=3.184466
low  count=4  mean_pair_normalized_nctc=1.570262
```

block size 2：

```text
task_summary
gsm8k     count=26 mean_pair_normalized_nctc=4.030950
hellaswag count=8  mean_pair_normalized_nctc=2.018066
humaneval count=27 mean_pair_normalized_nctc=6.202091
lambada   count=2  mean_pair_normalized_nctc=1.381958
math      count=20 mean_pair_normalized_nctc=3.348492

dependency_group_summary
high count=73 mean_pair_normalized_nctc=4.647000
low  count=10 mean_pair_normalized_nctc=1.890845
```

逐 block 记录已保存：

```text
outputs/qwen3_0_6b_block_nctc.jsonl
outputs/qwen3_0_6b_block2_nctc.jsonl
```

结论：在 block-level / multi-order 协议下，Qwen3-0.6B 的 high dependency
组 nCTC 高于 low dependency 组。这比早先整段 AR-only proxy 更符合论文语言侧结论。

### LLaDA / Dream preflight

已通过 Hugging Face 代理检查：

```text
GSAI-ML/LLaDA-8B-Instruct config/tokenizer OK
Dream-org/Dream-v0-Instruct-7B config/tokenizer OK
```

进一步尝试加载 LLaDA-8B 权重时，权重下载成功，但模型加载失败：

```text
AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
```

该错误更像是当前 `transformers==5.12.1` 与 LLaDA remote code 不兼容，
而不是显存不足。LLaDA 官方通常要求较旧的 `transformers` 版本；因此真实 dLLM
复现建议单独建立环境，例如：

```text
transformers==4.38.2
torch==2.7.1+cu128 或 LLaDA/DepCap 推荐版本
```

然后在外部 LLaDA/DepCap repo 中运行 fixed/adaptive decoding benchmark。

已创建独立环境：

```text
/gpfs/hulab/liyongqi/rl/external/LLaDA
torch 2.7.1+cu128
transformers 4.38.2
```

在该环境中，LLaDA-8B 已成功完成三个 smoke：

```text
load-only: OK, CUDA memory allocated 14.93 GB
generation: OK, "Tom has 5 apples."
likelihood: OK, produced two small log-likelihood records
curated likelihood: OK, produced ten benchmark-style log-likelihood records
fixed/adaptive generation: OK, produced thirty generation records
confidence-gated adaptive: ran, but naive mean top1 confidence did not improve over fixed baselines
```

输出文件：

```text
outputs/llada_generate_smoke.json
outputs/llada_likelihood_smoke.jsonl
outputs/llada_benchmark_likelihood_smoke.jsonl
outputs/llada_fixed_adaptive_smoke.jsonl
outputs/llada_confidence_adaptive_smoke.jsonl
```

curated likelihood 汇总：

```text
high count=6 mean_log_likelihood=-12.390630
low  count=4 mean_log_likelihood=-3.483023
```

该 likelihood 是 raw sequence log-likelihood，high 组 completion 更长、更结构化，
所以更负不能直接解释为“更差”或“依赖更强”。

fixed/adaptive generation smoke 汇总：

```text
adaptive_task_gated count=10 mean_f1=0.650485 mean_seconds=1.091122
fixed_b16           count=10 mean_f1=0.637152 mean_seconds=1.090787
fixed_b8            count=10 mean_f1=0.568667 mean_seconds=1.141657
```

`adaptive_task_gated` 是任务标签驱动的轻量 proxy：high-dep 用 block 8，
low-dep 用 block 16。因此它验证了真实 LLaDA 模型中 block granularity 会影响输出，
但还不是论文中基于 token uncertainty 的动态 adaptive decoding。

confidence-gated adaptive 进一步尝试使用全 mask response 的 mean top1 confidence
选择 block：

```text
threshold=0.42
adaptive_confidence count=10 mean_f1=0.568667
fixed_b16           count=10 mean_f1=0.637152
fixed_b8            count=10 mean_f1=0.568667
```

诊断：

```text
high mean_conf=0.344279 blocks=[8, 8, 8, 8, 8, 8]
low  mean_conf=0.358154 blocks=[8, 8, 8, 16]
```

这是一个负结果：naive mean top1 confidence 没有可靠地区分 high/low，
导致 low 组多数仍使用小 block。离线 threshold sweep 显示，0.26-0.32
附近的结果接近 fixed_b16，0.40-0.44 则退化到 fixed_b8。后续需要替换为
entropy、top1-top2 margin、低置信 token 比例，或直接接 DepCap/Fast-dLLM 的策略。

### 当前 paper-close 程度

更新后的判断：

```text
工程支架：约 80% 完成
语言侧 nCTC 实证：约 60% 完成
dLLM fixed/adaptive 实证：LLaDA fixed/adaptive smoke 完成；naive confidence adaptive 失败，需更强 uncertainty probe
diffusion/flow 实证：toy surrogate 完成，真实模型未完成
完整 paper-close 复现：仍需 LLaDA/Dream 独立环境评测
```

## 2026-06-22 动态 adaptive 解码准备

已在外部 LLaDA 环境中新增 dynamic adaptive 脚本：

```text
/gpfs/hulab/liyongqi/rl/external/LLaDA/scripts/mindrl_llada_dynamic_adaptive.py
```

该脚本实现了更接近论文 Appendix G 的 decoding loop：

```text
1. 初始化固定长度 response canvas。
2. 每轮对所有未提交 mask 位置前向。
3. 计算 token-level uncertainty：
   - top1: 1 - p_top1
   - margin: 1 - (p_top1 - p_top2)
   - entropy: normalized entropy
4. 用 B_t = floor(alpha / (mean_uncertainty + eps)) 动态选择 block size。
5. 用距离约束 gap=4 从高置信候选里选择提交位置。
6. 对比 fixed_b8_dynamic_loop、fixed_b16_dynamic_loop、adaptive_dynamic。
```

已完成语法检查：

```text
.venv/bin/python -m py_compile scripts/mindrl_llada_dynamic_adaptive.py
```

尚未运行的原因：

```text
2026-06-22 01:58 GPU4-7 均被 VLLM::Worker 占用约 23GB 显存；
LLaDA-8B 需要约 15GB 显存，当前无法安全加载。
```

GPU 空闲后建议运行：

```bash
cd /gpfs/hulab/liyongqi/rl/external/LLaDA

CUDA_VISIBLE_DEVICES=6 \
HF_HOME="/gpfs/hulab/liyongqi/.cache/huggingface" \
HF_HUB_DISABLE_XET=1 \
http_proxy="http://10.11.0.51:7890" \
https_proxy="http://10.11.0.51:7890" \
HTTP_PROXY="http://10.11.0.51:7890" \
HTTPS_PROXY="http://10.11.0.51:7890" \
.venv/bin/python scripts/mindrl_llada_dynamic_adaptive.py \
  --max-examples 10 \
  --gen-length 32 \
  --max-steps 16 \
  --uncertainty top1 \
  --alpha 4.0 \
  --gap 4
```

如果 top1 结果不理想，下一组应跑：

```bash
.venv/bin/python scripts/mindrl_llada_dynamic_adaptive.py \
  --max-examples 10 \
  --gen-length 32 \
  --max-steps 16 \
  --uncertainty margin \
  --alpha 4.0 \
  --gap 4

.venv/bin/python scripts/mindrl_llada_dynamic_adaptive.py \
  --max-examples 10 \
  --gen-length 32 \
  --max-steps 16 \
  --uncertainty entropy \
  --alpha 4.0 \
  --gap 4
```

预期输出：

```text
outputs/llada_dynamic_adaptive_smoke.jsonl
```

该实验是下一步最关键的 dLLM 复现任务：它会把当前的
task-gated / naive confidence-gated adaptive，推进到真正逐步更新的
token-level uncertainty adaptive decoding。

### 2026-06-23 dynamic adaptive 运行结果

后台 watcher 已在 GPU0 上完成三组 uncertainty：

```text
outputs/llada_dynamic_adaptive_top1.jsonl
outputs/llada_dynamic_adaptive_margin.jsonl
outputs/llada_dynamic_adaptive_entropy.jsonl
outputs/logs/llada_dynamic_adaptive_top1.log
outputs/logs/llada_dynamic_adaptive_margin.log
outputs/logs/llada_dynamic_adaptive_entropy.log
```

运行设置：

```text
model: GSAI-ML/LLaDA-8B-Instruct
max_examples: 10
gen_length: 32
max_steps: 16
alpha: 4.0
gap: 4
modes: fixed_b8_dynamic_loop, fixed_b16_dynamic_loop, adaptive_dynamic
```

整体结果：

```text
top1:
  adaptive_dynamic mean_f1=0.506464 mean_seconds=0.209433 avg_block=5.3181 actual_steps=6.1
  fixed_b16         mean_f1=0.440755 mean_seconds=0.198240 avg_block=5.5771 actual_steps=5.8
  fixed_b8          mean_f1=0.440755 mean_seconds=0.511259 avg_block=5.5771 actual_steps=5.8

margin:
  adaptive_dynamic mean_f1=0.506464 mean_seconds=0.212778 avg_block=5.2114 actual_steps=6.2
  fixed_b16         mean_f1=0.391580 mean_seconds=0.200037 avg_block=5.5467 actual_steps=5.8
  fixed_b8          mean_f1=0.391580 mean_seconds=0.256558 avg_block=5.5467 actual_steps=5.8

entropy:
  adaptive_dynamic mean_f1=0.418716 mean_seconds=0.195341 avg_block=5.6533 actual_steps=5.7
  fixed_b16         mean_f1=0.418716 mean_seconds=0.195697 avg_block=5.6533 actual_steps=5.7
  fixed_b8          mean_f1=0.418716 mean_seconds=0.474718 avg_block=5.6533 actual_steps=5.7
```

按依赖组：

```text
top1:
  high adaptive=0.526647 fixed=0.530426
  low  adaptive=0.476190 fixed=0.306250

margin:
  high adaptive=0.526647 fixed=0.519300
  low  adaptive=0.476190 fixed=0.200000

entropy:
  high adaptive=0.527027 fixed=0.527027
  low  adaptive=0.256250 fixed=0.256250
```

解释：

```text
1. top1 / margin dynamic adaptive 在 overall mean F1 上优于当前 dynamic-loop fixed baselines。
2. 改善主要来自 low 组；high 组与 fixed 接近。
3. entropy 在当前归一化和 alpha 下没有产生有效差异。
4. fixed_b8 与 fixed_b16 在该 32-token canvas + gap=4 设置下实际 average block size 很接近，
   因此这个结果还不是论文表格中的 compute-neutral fixed-B=8 vs adaptive 对比。
```

当前判断：

```text
dynamic token-level adaptive 路径已经跑通；
top1/margin 是当前最有希望继续扩展的 uncertainty proxy；
下一步需要做 alpha grid、128-token canvas、compute-neutral calibration 和 official metric。
```

## 2026-06-22 后台 GPU watcher 与 flow/VLA 实现

### GPU watcher

已在外部 LLaDA 环境中新增并启动后台 watcher：

```text
/gpfs/hulab/liyongqi/rl/external/LLaDA/scripts/wait_and_run_dynamic_adaptive.sh
```

行为：

```text
1. 每 300 秒检查 GPU4-7。
2. 若任一 GPU 空闲显存 >= 17000 MiB，则选择该 GPU。
3. 依次运行 top1 / margin / entropy 三组 dynamic adaptive decoding。
4. 输出到：
   outputs/llada_dynamic_adaptive_top1.jsonl
   outputs/llada_dynamic_adaptive_margin.jsonl
   outputs/llada_dynamic_adaptive_entropy.jsonl
5. 日志输出到 outputs/logs/。
```

启动时 GPU 仍被 `VLLM::Worker` 占用，watcher 已进入等待循环。

### Flow/diffusion interface

新增模块：

```text
src/mindrl/flow_diffusion_interface.py
examples/run_flow_diffusion_interface_pilot.py
tests/test_flow_diffusion_interface.py
```

实现内容：

```text
FlowDiffusionTrace
  -> summarize_flow_trace
  -> flow_barrier_profile
  -> MindRLController decision
  -> risk_adjusted_reward
```

轻量示例结果：

```text
trace       strategy        adapter                  risk_adjusted_reward
low_drift   score_routing   score_routing_only        0.3889
low_drift   barrier_gated   anchored_flow_surrogate   0.8428
high_drift  score_routing   score_routing_only       -0.2652
high_drift  barrier_gated   anchored_flow_surrogate  -0.1185
```

解释：在高 drift / 高 surrogate variance trace 上，barrier-gated controller
通过增强 anchor、收紧 clip、降低 branch weight 改善 risk-adjusted reward。

### VLA / AR+flow interface

新增模块：

```text
src/mindrl/vla_interface.py
examples/run_vla_interface_pilot.py
tests/test_vla_interface.py
```

实现内容：

```text
ARBranchTrace + FlowDiffusionTrace + semantic_staleness + world_uncertainty
  -> AR exact_ratio decision
  -> flow anchored surrogate decision
  -> semantic_refresh_budget
  -> risk_adjusted_score
```

轻量示例结果：

```text
strategy             ar_adapter   flow_adapter              semantic_refresh  risk_adjusted_score
score_routing_only   exact_ratio  score_routing_only        0.000             0.2126
barrier_gated        exact_ratio  anchored_flow_surrogate   0.350             0.3459
```

解释：这还不是 LIBERO/EO1 真实验，但已经把论文中的
AR exact credit、flow anchored surrogate、semantic/world uncertainty barrier
组织成可测试接口。

### 新增验证

```text
uv run python -m unittest tests.test_flow_diffusion_interface tests.test_vla_interface
Ran 5 tests
OK
```

下一步：

```text
1. 等 GPU watcher 产出 dynamic adaptive top1/margin/entropy 结果。
2. 将 dynamic adaptive 结果与当前 fixed/adaptive smoke 对比。
3. 若 margin/entropy 有改善，再做 alpha grid 与 compute-neutral calibration。
4. flow/VLA 侧继续接真实 diffusers / UniRL / LIBERO 数据源。
```
