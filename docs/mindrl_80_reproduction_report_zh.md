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
cd /gpfs/hulab/liyongqi/rl/mindrl_repo
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
