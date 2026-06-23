# MINDRL 复现模型可行性评估

## 当前机器环境

当前可用环境记录：

- Linux + CUDA 机器可见 8 张 RTX 3090，但最近检查时显存基本被其他任务占满。
- `torch` / `transformers` 已在 `uv` 环境中可用。
- Hugging Face 在线下载路径可能不稳定；推荐优先使用 `--local-files-only` 检查本地 cache。
- 第一阶段不抢 GPU，先跑 CPU/mock 或 CPU/cache pilot；GPU 空闲后再切到 CUDA。

已验证：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
uv run python examples/offline_random_ar_proxy_nctc.py
uv run python examples/run_task_nctc_proxy.py --sample preset --max-examples 3 --device cpu --mock-scorer
```

## 模型候选排序

### 1. `sshleifer/tiny-gpt2`

用途：最小真实 Hugging Face causal LM smoke。

优点：

- 权重很小。
- 适合验证 `hf_ar_proxy_nctc.py` 和 `run_task_nctc_proxy.py` 的真实模型加载、tokenization、logprob scoring。

缺点：

- 模型太弱，不能代表论文任务趋势。
- 当前网络下载卡住，需要 Hugging Face cache 或更稳定网络。

结论：如果网络恢复，这是第一优先级。

### 2. `Qwen/Qwen3-0.6B`

用途：较有意义的 AR scorer，用于 teacher-forced nCTC proxy。

优点：

- 与论文 Qwen3-8B 同系列。
- 0.6B 在 CPU 小样本或单张 3090 上比 8B 可行得多。
- 对 GSM8K / HumanEval 小样本更有意义。

缺点：

- 仍需下载数百 MB 到 1GB 级权重。
- 只能做 AR scorer proxy，不能复现 paired dLLM decoding。

结论：真实低成本 AR scorer 的最佳候选。

### 3. `Qwen/Qwen3-1.7B`

用途：比 0.6B 更稳定的 scorer。

优点：

- 推理质量更好。
- GPU 空闲时仍可能单卡运行。

缺点：

- 下载和推理成本更高。

结论：如果 0.6B 跑通且趋势不稳定，再尝试。

### 4. LLaDA / Dream / d3LLM 系列

用途：更接近论文的 dLLM / diffusion LM 复现。

优点：

- 模型类型更接近论文 parallel discrete / dLLM 设置。
- 可评估 fixed block vs adaptive block。

缺点：

- 公开主力模型多为 8B BF16。
- 通常需要专用 repo、remote code、较大显存。
- 不适合作为第一轮 CPU pilot。

结论：不适合作为本机第一轮；适合放到 GPU 机器上做 paper-close 或 extended reproduction。

## 当前已实现复现层级

### A. Theory / toy reproduction

已完成：

- CTC 随依赖增强上升。
- block refinement 降低 within-block factorization barrier。
- adaptive block size 随 uncertainty 增大而缩小。

### B. Offline AR scoring path

已完成：

- `examples/offline_random_ar_proxy_nctc.py`
- 随机 tiny GPT-2 初始化，不下载权重。
- 验证 AR logprob scoring 与 dependency-gap pipeline。

### C. Hugging Face causal LM path

已实现但未跑通下载：

- `examples/hf_ar_proxy_nctc.py`
- `examples/run_task_nctc_proxy.py`
- `src/mindrl/hf_scorer.py`

推荐命令：

```bash
uv run python examples/hf_ar_proxy_nctc.py --model sshleifer/tiny-gpt2
uv run python examples/hf_ar_proxy_nctc.py --model Qwen/Qwen3-0.6B
uv run python examples/run_task_nctc_proxy.py --model sshleifer/tiny-gpt2 --device cpu --sample preset --max-examples 3 --local-files-only
MODEL=Qwen/Qwen3-0.6B MAX_EXAMPLES=20 bash examples/run_gpu_pilot.sh
```

若已手动下载到 Hugging Face cache，可以加网络环境后直接运行。

### D. Task-level nCTC proxy pilot

已新增：

- `src/mindrl/benchmark_tasks.py`
- `examples/run_task_nctc_proxy.py`
- `examples/run_cpu_pilot.sh`
- `examples/run_gpu_pilot.sh`

CPU 小规模路径：

```bash
uv run python examples/run_task_nctc_proxy.py --sample preset --max-examples 3 --device cpu --mock-scorer
bash examples/run_cpu_pilot.sh
```

`run_cpu_pilot.sh` 和 `run_gpu_pilot.sh` 会优先使用 `uv run python`；如果当前 shell 没有 `uv`，会回退到 `PYTHONPATH=src python3`。

GPU 扩展路径：

```bash
MODEL=Qwen/Qwen3-0.6B MAX_EXAMPLES=20 bash examples/run_gpu_pilot.sh
```

`--mock-scorer` 只用于验证数据流、JSONL 输出和摘要聚合；真实趋势判断应使用 Hugging Face scorer。

## 与论文 protocol 的差距

当前 AR proxy 做的是：

```text
joint = log p_AR(completion block | prompt)
marginal = sum_i log p_AR(token_i | prompt)
```

论文 nCTC 做的是：

```text
joint = chain-rule log p_ref(x_M | C)
marginal = sum_i log p_ref(x_i | C)
```

论文的 `C` 是 masked/dLLM visibility context，paired AR scorer 与 dLLM checkpoint 对齐。当前 proxy 只能验证 dependency signal pipeline，不能声称复现论文表 1。

## 下一步建议

1. 先解决模型下载/cache：
   - 优先 `sshleifer/tiny-gpt2` 验证真实 HF path。
   - 然后 `Qwen/Qwen3-0.6B` 做小样本 AR scorer。

2. 做小样本 nCTC proxy：
   - 先用内置 `gsm8k_smoke`、`humaneval_smoke`、`hellaswag_smoke`、`lambada_smoke` 验证流程。
   - 再通过 JSONL 输入扩展到 GSM8K 50 条、HumanEval 50 条、HellaSwag 50 条。
   - 比较 high-dep vs low-dep 的 dependency gap。

3. 如果要更接近论文：
   - 上 GPU 机器。
   - 选 LLaDA/Dream/d3LLM 或按论文 AR->dLLM conversion。
   - 实现 fixed `B=8` vs adaptive schedule。

## 结论

本机最容易跑的真实模型是：

```text
sshleifer/tiny-gpt2 -> Qwen/Qwen3-0.6B -> Qwen/Qwen3-1.7B
```

最接近论文但不适合本机首轮的是：

```text
LLaDA / Dream / d3LLM 8B
```

因此，当前建议先用 `Qwen3-0.6B` 复现 AR scorer nCTC proxy，再把 dLLM adaptive decoding 留给 GPU 环境。
