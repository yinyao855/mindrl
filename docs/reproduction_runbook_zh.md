# MINDRL 复现运行说明

本文档记录当前仓库如何运行，以及当前小规模实验是否支持
`Reward Is Not a Universal Interface for Generative Reinforcement Learning`
的主要结论。

## 1. 当前复现范围

当前仓库已经覆盖的是论文中最容易低成本验证的语言离散侧机制：

- toy CTC / total correlation barrier 验证；
- nCTC / pair-normalized dependency gap 计算工具；
- inverse-uncertainty adaptive block size；
- Hugging Face causal LM teacher-forcing scorer；
- task-level AR scorer proxy，用来粗略检查不同任务 completion block 的依赖强度。

当前仓库还没有完整复现论文中更重的部分：

- paired AR-to-dLLM nCTC protocol；
- dLLM fixed block vs adaptive block decoding；
- diffusion / flow policy 的 reward-to-update surrogate；
- AR + flow / VLA / embodied policy 的多分支 controller 实验。

因此当前结果应称为 **低成本语言侧 proxy reproduction**，不能直接等同于论文完整复现。

## 2. 环境准备

项目使用 `uv` 管理环境。当前 `pyproject.toml` 已配置：

```toml
[tool.uv]
cache-dir = "/gpfs/hulab/liyongqi/.cache/uv"
```

这样在不同集群节点上同步环境时，依赖 cache 会放在 GPFS 下。

如果当前节点没有 `uv`，可以先安装：

```bash
mkdir -p /home/liyongqi/.local/bin
http_proxy="http://10.11.0.51:7890" \
https_proxy="http://10.11.0.51:7890" \
python3 -m pip install --user uv
```

然后进入项目目录同步环境：

```bash
cd /gpfs/hulab/liyongqi/rl/mindrl_repo
PATH="/home/liyongqi/.local/bin:$PATH" \
http_proxy="http://10.11.0.51:7890" \
https_proxy="http://10.11.0.51:7890" \
uv sync --index-strategy unsafe-best-match
```

当前验证过的关键版本：

```text
torch == 2.7.1+cu128
CUDA driver == 12.8 compatible
GPU == NVIDIA GeForce RTX 3090
```

## 3. 基础验证

运行单元测试：

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python -m unittest discover -s tests -p "test_*.py"
```

当前结果：

```text
Ran 18 tests
OK
```

运行无需下载模型的 toy / offline 示例：

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/minimal_discrete_repro.py

PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/offline_random_ar_proxy_nctc.py
```

这一步主要验证代码路径，不代表真实模型趋势。

## 4. GPU 与代理检查

查看 GPU：

```bash
nvidia-smi
```

当前节点上 GPU 4、5、7 基本空闲时，可以选 GPU 4：

```bash
CUDA_VISIBLE_DEVICES=4 \
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0], device="cuda")
print(float(x.sum().cpu()))
PY
```

已验证输出包括：

```text
torch 2.7.1+cu128
cuda_available True
device_name NVIDIA GeForce RTX 3090
sum 3.0
```

`~/.bashrc` 中有 `proxy_on` 函数，但它位于非交互 shell 的 return 之后，脚本运行时不会自动生效。因此建议在命令里显式设置代理：

```bash
http_proxy="http://10.11.0.51:7890"
https_proxy="http://10.11.0.51:7890"
HTTP_PROXY="http://10.11.0.51:7890"
HTTPS_PROXY="http://10.11.0.51:7890"
```

## 5. 小数据集运行

仓库提供了一个小规模 JSONL：

```text
examples/data/mindrl_small_proxy.jsonl
```

它包含 5 条 high dependency 样本和 5 条 low dependency 样本。该数据只用于 smoke / pilot，不是正式 benchmark。

先跑 mock scorer，检查数据读取与聚合：

```bash
PATH="/home/liyongqi/.local/bin:$PATH" \
uv run python examples/run_task_nctc_proxy.py \
  --sample jsonl \
  --jsonl-path examples/data/mindrl_small_proxy.jsonl \
  --device cpu \
  --mock-scorer \
  --max-examples 10
```

当前 mock 结果：

```text
dependency_group_summary
high count=5 mean_pair_normalized_gap=0.052000
low  count=5 mean_pair_normalized_gap=0.010000
```

mock scorer 是人为构造的 sanity check，因此它只说明 pipeline 能区分标注组，不说明真实模型结论。

## 6. Qwen3-0.6B GPU 运行

使用真实 Hugging Face causal LM：

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
  --jsonl-path examples/data/mindrl_small_proxy.jsonl \
  --max-examples 10 \
  --max-block-tokens 8
```

当前真实模型结果：

```text
task_summary
code_condition_01     mean_pair_normalized_gap=4.148805
code_nested_loop_01   mean_pair_normalized_gap=3.088493
logic_order_01        mean_pair_normalized_gap=2.663426
math_multistep_01     mean_pair_normalized_gap=2.498383
math_multistep_02     mean_pair_normalized_gap=3.184187
simple_completion_01  mean_pair_normalized_gap=11.101562
simple_event_01       mean_pair_normalized_gap=1.637573
simple_event_02       mean_pair_normalized_gap=3.170605
simple_fact_01        mean_pair_normalized_gap=7.242188
simple_fact_02        mean_pair_normalized_gap=5.171875

dependency_group_summary
high count=5 mean_pair_normalized_gap=3.116659
low  count=5 mean_pair_normalized_gap=5.664761
```

## 7. 结果是否符合论文结论

当前分层判断如下。

### 已支持的结论

toy / analytic 层面支持论文的离散侧直觉：

- 变量依赖越强，CTC 越高；
- 将大 block refine 成小 block 会降低 within-block factorization barrier；
- uncertainty 越高，自适应 block size 越小。

这些对应论文中 “parallel discrete block 存在依赖障碍，不能无脑并行更新” 的核心机制。

### 尚未被真实模型小实验支持的结论

Qwen3-0.6B 的小数据集 AR proxy 没有得到 “high dependency 任务 gap 高于 low dependency 任务” 的趋势，当前结果反而是 low 组更高。

主要原因有三点：

1. 当前指标是 AR-only proxy，不是论文的 paired AR-to-dLLM nCTC。
2. low 组的短 completion 在 Qwen tokenizer 下可能被切成多个强相关 token，导致 prompt-only marginal 很低、chain joint 很高。
3. 小数据集只有 10 条样本，任务和 completion 长度没有做严格配平。

因此，这一轮真实模型实验说明：

```text
GPU/HF/Qwen scoring pipeline 已跑通；
但当前小数据集 AR proxy 尚不能复现论文任务趋势。
```

## 8. 是否已经复现论文

严格来说：**还没有完整复现论文**。

当前已经完成的是：

- 论文离散侧机制的 toy reproduction；
- AR scorer proxy 的工程路径；
- Qwen3-0.6B 单卡 GPU 小样本运行。

还需要继续做的 paper-close reproduction：

1. 使用真实 benchmark 样本：
   - GSM8K / MATH / HumanEval 作为 high dependency；
   - HellaSwag / LAMBADA 作为 low dependency；
   - 每类至少 50-100 条，控制 completion token 数。

2. 改进 nCTC measurement：
   - 不只用完整 completion 前缀；
   - 按论文方式采样 block；
   - 对同一 block 计算多种 order 的 joint chain logprob；
   - 对 marginal context 做更接近 masked / dLLM visibility context 的构造。

3. 上 dLLM / parallel discrete 模型：
   - 复现 fixed block；
   - 复现 adaptive block；
   - 比较 high-dep 和 low-dep 任务上的 gain。

4. 如果要覆盖论文全部模型范围，还要增加：
   - diffusion / flow policy surrogate；
   - AR + flow / VLA 分支；
   - branch-level controller ablation。

## 9. 下一步建议

短期建议先做语言侧 paper-close pilot：

```text
Qwen3-0.6B / 1.7B AR scorer
  -> benchmark JSONL 50-100 条/任务
  -> block sampling + multi-order nCTC estimate
  -> high-dep vs low-dep trend
```

如果这个趋势稳定，再投入更贵的 dLLM adaptive decoding。

如果趋势仍不稳定，应优先检查 metric protocol，而不是直接换大模型；因为当前失败点更像是 proxy definition 和样本构造问题，不是 GPU 或环境问题。
