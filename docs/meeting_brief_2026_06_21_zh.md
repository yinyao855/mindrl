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
dLLM fixed/adaptive：LLaDA load/generation/likelihood/curated likelihood smoke 完成，fixed-vs-adaptive benchmark 未完成
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

1. 把 LLaDA likelihood smoke 扩展到 curated benchmark 全部样本。
2. 接 DepCap / Fast-dLLM，实现真实 fixed block vs adaptive block 对比。
3. 将 LLaDA 结果与 Qwen block-level nCTC 分组关联，验证 high-dep 上 adaptive gain 是否更明显。
4. 若结果稳定，再扩大到真实 GSM8K/HumanEval/HellaSwag/LAMBADA 子集。

建议汇报时强调：

- 目前不是“已经完整复现论文”，而是“语言侧核心机制已初步复现，LLaDA 真实模型路径已跑通”。
- 真正的 paper-close 复现关键在下一步 LLaDA/Dream fixed-vs-adaptive benchmark。
