# MINDRL 论文复现进展临时汇报

日期：2026-06-21

## 一句话总结

目前已经完成 MINDRL 复现的工程支架，并在 Qwen3-0.6B 上跑通
block-level multi-order nCTC。语言侧结果支持论文核心判断：
**高依赖任务的 token/block 依赖更强，nCTC 明显高于低依赖任务**。

dLLM 侧已经定位并解决了 LLaDA 加载失败的环境问题：需要单独使用
`transformers==4.38.2`，不能复用主项目的 `transformers==5.12.1` 环境。
目前 LLaDA-8B 已完成 load-only、generation smoke 和 likelihood smoke。

## 当前已完成工作

### 1. 复现代码仓库

主项目：

```text
/gpfs/hulab/liyongqi/rl/mindrl_repo
```

已提交的关键 commit：

```text
d52b809 Add MINDRL 80 percent reproduction scaffold
fd4bd98 Record Qwen block-level nCTC reproduction results
```

当前主项目包含：

- CTC / nCTC toy barrier 验证。
- block-level multi-order nCTC measurement。
- Hugging Face causal LM scorer。
- curated benchmark JSONL，覆盖 GSM8K / MATH / HumanEval / HellaSwag / LAMBADA 风格样本。
- dLLM fixed/adaptive block 的 mock adapter。
- controller ablation。
- toy flow surrogate。
- 中文复现说明和阶段报告。

### 2. 测试情况

完整单元测试已通过：

```text
Ran 31 tests
OK
```

说明当前工程支架是可重复运行的，不只是一次性 notebook/script。

## 关键实验结果

### Qwen3-0.6B block-level nCTC

模型：

```text
Qwen/Qwen3-0.6B
```

设备：

```text
RTX 3090, CUDA 12.8, torch 2.7.1+cu128
```

输出文件：

```text
outputs/qwen3_0_6b_block_nctc.jsonl
outputs/qwen3_0_6b_block2_nctc.jsonl
```

block size 4：

```text
high mean_pair_normalized_nctc = 3.184466
low  mean_pair_normalized_nctc = 1.570262
```

block size 2：

```text
high mean_pair_normalized_nctc = 4.647000
low  mean_pair_normalized_nctc = 1.890845
```

解释：

- 早期整段 AR-only proxy 曾出现 low 组偏高的问题。
- 升级为 block-level multi-order nCTC 后，high 组稳定高于 low 组。
- 这更接近论文关注的 block 内依赖障碍，也更支持论文语言侧结论。

当前结论：

```text
语言侧 nCTC 趋势：已得到初步支持。
```

## LLaDA / dLLM 进展

### 原问题

在主项目环境中直接加载 LLaDA-8B：

```text
transformers==5.12.1
```

会报错：

```text
AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
```

原因：

- LLaDA 使用 Hugging Face `trust_remote_code=True`。
- LLaDA 官方 remote code 依赖旧版 Transformers API。
- LLaDA README 明确建议 `transformers==4.38.2`。
- 因此该问题不是 GPU 显存不足，而是 Transformers 版本不兼容。

### 解决方式

已创建独立外部环境：

```text
/gpfs/hulab/liyongqi/rl/external/LLaDA
```

该环境版本：

```text
torch 2.7.1+cu128
transformers 4.38.2
cuda True
```

已完成 LLaDA-8B load-only preflight：

```text
model: GSAI-ML/LLaDA-8B-Instruct
model class: LLaDAModelLM
CUDA memory allocated: 14.93 GB
preflight: OK
```

说明：

```text
LLaDA 加载问题已经解决，后续可以在 external/LLaDA 环境继续做真实 dLLM 实验。
```

### LLaDA generation smoke

已在独立环境中运行极小生成任务：

```text
model: GSAI-ML/LLaDA-8B-Instruct
prompt: If Tom has 3 apples and buys 2 more, how many apples does he have?
completion: Tom has 5 apples.
steps: 16
gen_length: 16
block_length: 8
CUDA memory allocated: 14.94 GB
```

输出文件：

```text
outputs/llada_generate_smoke.json
```

结论：LLaDA 真实生成路径已经跑通，不再只是 load-only。

### LLaDA likelihood smoke

已运行 LLaDA 官方 `get_log_likelihood.py` 的小样本版本：

```text
model: GSAI-ML/LLaDA-8B-Base
mc_num: 4
batch_size: 2
```

结果：

```text
gsm8k_smoke     high log_likelihood = -8.064505
hellaswag_smoke low  log_likelihood = -6.714966
```

输出文件：

```text
outputs/llada_likelihood_smoke.jsonl
```

说明：这一步只验证 likelihood 路径可用，样本数和 MC 次数都太小，不能据此判断论文趋势。

### LLaDA curated benchmark likelihood

进一步在 curated benchmark 10 条样本上运行 LLaDA-8B-Base likelihood：

```text
model: GSAI-ML/LLaDA-8B-Base
mc_num: 4
batch_size: 2
```

输出文件：

```text
outputs/llada_benchmark_likelihood_smoke.jsonl
```

汇总：

```text
group_summary
high count=6 mean_log_likelihood=-12.390630
low  count=4 mean_log_likelihood=-3.483023

task_summary
gsm8k     count=2 mean_log_likelihood=-19.135222
math      count=2 mean_log_likelihood=-8.015770
humaneval count=2 mean_log_likelihood=-10.020898
hellaswag count=2 mean_log_likelihood=-3.324169
lambada   count=2 mean_log_likelihood=-3.641876
```

解释：high 组 completion 更长、更结构化，因此 raw log-likelihood 更负是预期现象。
该结果只说明 LLaDA likelihood evaluation path 已经可跑；若要用于论文结论，需要做
长度归一化、block-level nCTC 或 fixed/adaptive decoding 指标。

### LLaDA fixed/adaptive generation smoke

已在 curated benchmark 10 条样本上运行 LLaDA-8B-Instruct 生成对比：

```text
gen_length: 32
steps: 32
modes:
  fixed_b8
  fixed_b16
  adaptive_task_gated
```

其中 `adaptive_task_gated` 是轻量 proxy：high-dep 样本使用 block 8，
low-dep 样本使用 block 16。它不是论文的 token-level uncertainty adaptive，
但已经是同一个真实 LLaDA 模型上的 fixed/adaptive block 对比 smoke。

输出文件：

```text
outputs/llada_fixed_adaptive_smoke.jsonl
```

汇总：

```text
mode_summary
adaptive_task_gated count=10 mean_f1=0.650485 mean_seconds=1.091122
fixed_b16           count=10 mean_f1=0.637152 mean_seconds=1.090787
fixed_b8            count=10 mean_f1=0.568667 mean_seconds=1.141657

group_mode_summary
high adaptive_task_gated count=6 mean_f1=0.721397
high fixed_b16           count=6 mean_f1=0.699175
high fixed_b8            count=6 mean_f1=0.721397
low  adaptive_task_gated count=4 mean_f1=0.544118
low  fixed_b16           count=4 mean_f1=0.544118
low  fixed_b8            count=4 mean_f1=0.339572
```

解释：

- high 组中 adaptive 与 fixed_b8 等价，因此结果相同。
- low 组中 adaptive 与 fixed_b16 等价，因此避免了 fixed_b8 在短 completion 上的退化。
- 该 smoke 支持“按依赖强度选择 block granularity 有意义”这个方向，但还不能替代论文的动态 uncertainty-guided adaptive decoding。

### LLaDA confidence-gated adaptive smoke

进一步尝试了一个更接近论文方向的 adaptive proxy：先对全 mask response 做一次前向，
计算 response 区域的 mean top-1 confidence；低 confidence 用 block 8，高 confidence 用 block 16。

运行设置：

```text
model: GSAI-ML/LLaDA-8B-Instruct
gen_length: 32
steps: 32
confidence_threshold: 0.42
modes:
  fixed_b8
  fixed_b16
  adaptive_confidence
```

输出文件：

```text
outputs/llada_confidence_adaptive_smoke.jsonl
```

汇总：

```text
mode_summary
adaptive_confidence count=10 mean_f1=0.568667 mean_seconds=1.088860
fixed_b16           count=10 mean_f1=0.637152 mean_seconds=1.088670
fixed_b8            count=10 mean_f1=0.568667 mean_seconds=1.158554

adaptive_confidence_diagnostics
high mean_conf=0.344279 blocks=[8, 8, 8, 8, 8, 8]
low  mean_conf=0.358154 blocks=[8, 8, 8, 16]
```

结论：这个 naive confidence probe 是一个负结果。它没有可靠地区分 high/low 任务，
导致 low 组多数仍选择 block 8，整体表现退化到 fixed_b8 水平。

离线 threshold sweep 显示，较低阈值 0.26-0.32 的 mean F1 为 0.637152，
等价或接近大多数样本使用 block 16；阈值 0.40-0.44 则退化到 0.568667。
这说明简单 mean top-1 confidence 不是足够好的 adaptive block probe。

下一步应改用更细粒度的 token/block uncertainty，例如：

- 每轮低置信 token 比例；
- top1-top2 margin；
- entropy；
- 与 Qwen block-level nCTC 关联的 block risk；
- DepCap / Fast-dLLM 中已有的 adaptive block 策略。

### 2026-06-22 动态 adaptive 脚本准备情况

已新增脚本：

```text
/gpfs/hulab/liyongqi/rl/external/LLaDA/scripts/mindrl_llada_dynamic_adaptive.py
```

该脚本实现了逐步更新版 adaptive decoding：

```text
每轮 forward
  -> 对所有 uncommitted mask positions 计算 uncertainty
  -> B_t = floor(alpha / (mean_uncertainty + eps))
  -> clip 到 [B_min, B_max]
  -> 用 gap=4 的距离约束选择高置信位置提交
```

支持三种 uncertainty：

```text
top1   = 1 - p_top1
margin = 1 - (p_top1 - p_top2)
entropy = normalized entropy
```

已完成脚本语法检查，但尚未运行，原因是当前 GPU4-7 被 `VLLM::Worker`
占用约 23GB 显存，LLaDA-8B 无法安全加载。

GPU 空闲后应优先运行：

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

这一步是从 smoke 走向论文 Appendix G dynamic adaptive decoding 的关键实验。

重现命令：

```bash
cd /gpfs/hulab/liyongqi/rl/external/LLaDA

CUDA_VISIBLE_DEVICES=5 \
HF_HOME="/gpfs/hulab/liyongqi/.cache/huggingface" \
HF_HUB_DISABLE_XET=1 \
http_proxy="http://10.11.0.51:7890" \
https_proxy="http://10.11.0.51:7890" \
HTTP_PROXY="http://10.11.0.51:7890" \
HTTPS_PROXY="http://10.11.0.51:7890" \
.venv/bin/python scripts/mindrl_llada_preflight.py
```

## 当前复现程度判断

当前状态：

```text
工程支架：约 80% 完成
语言侧 nCTC 实证：约 60% 完成
dLLM fixed/adaptive：LLaDA load/generation/likelihood/fixed-adaptive smoke 完成，动态 adaptive benchmark 未完成
diffusion/flow：toy surrogate 完成，真实模型未完成
VLA/embodied：尚未进入真实任务
```

如果按论文全部模型范围衡量，还不能说已经完整复现。

如果按“主要机制是否跑通”衡量，目前已经完成：

1. nCTC 工具链。
2. 高依赖 vs 低依赖语言任务趋势。
3. dLLM 环境障碍定位与解决，并跑通 LLaDA generation / likelihood smoke。
4. controller/flow 的轻量 ablation 支架。

## 对论文结论的当前支持

已支持：

- reward 不是直接可通用于所有生成分支的更新接口。
- parallel/block discrete 生成存在可测量的依赖障碍。
- block-level nCTC 可以区分高依赖与低依赖任务。
- adaptive block 的方向是合理的：高不确定/高依赖时应缩小 block。

尚未支持或仍需真实实验：

- LLaDA/Dream 上 fixed block vs adaptive block 的真实 benchmark gain。
- diffusion/flow policy 的真实 reward-risk tradeoff。
- AR + flow / VLA 多分支 controller 的真实闭环效果。

## 下一步计划

短期优先级：

1. GPU 空闲后运行 `mindrl_llada_dynamic_adaptive.py` 的 top1/margin/entropy 三组。
2. 根据结果做 alpha grid / compute-neutral calibration。
3. 接 DepCap / Fast-dLLM，实现论文更接近的 adaptive block decoding。
4. 将 LLaDA 结果与 Qwen block-level nCTC 分组关联，验证 high-dep 上 adaptive gain 是否更明显。
5. 若结果稳定，再扩大到真实 GSM8K/HumanEval/HellaSwag/LAMBADA 子集。

建议汇报时强调：

- 目前不是“已经完整复现论文”，而是“语言侧核心机制已初步复现，LLaDA 真实模型路径已跑通”。
- 真正的 paper-close 复现关键在下一步 LLaDA/Dream fixed-vs-adaptive benchmark。
