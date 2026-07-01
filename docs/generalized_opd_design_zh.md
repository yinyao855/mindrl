# 广义 OPD 设计文档

## 目标

把 OPD 从“AR token-level teacher distillation”抽象成更通用的 **teacher-guided on-policy update**，让 MindRL 可以在不同 policy branch 上复用同一套控制流程：

```text
student 自己采样状态
teacher 在 student 状态上给局部监督
objective 使用该 branch 原生可更新对象计算 loss
report 暴露 raw / clipped / entropy / drift 等诊断
```

第一版目标不是接入所有真实大模型，而是建立稳定接口：

- AR：能把现有 `RolloutBatch` / `TeacherSignal` 映射到通用 OPD 状态和信号。
- diffusion / flow：能用 mock score / denoising / velocity 向量验证同一抽象可用。
- dLLM：暂不实现真实训练，只保留 block / nCTC 方向作为后续扩展。

## 非目标

- 不重写现有 `opd.py` 和 `peft_trainer.py`。
- 不在第一版支持真实 LLaDA / Dream 训练。
- 不做 14B multi-GPU PEFT。
- 不引入 diffusers 依赖。

## 核心抽象

### `OnPolicyState`

表示 student 自己采样或访问到的状态。

```text
state_id: str
branch: str
payload: tuple[float, ...]
metadata: dict
```

对不同分支，`payload` 的语义不同：

- AR：sampled token logprobs 或 token-level student score。
- diffusion：student denoising prediction / score vector。
- flow：student velocity / path surrogate vector。
- dLLM：block-level score vector。

### `TeacherGuidance`

表示 teacher 在同一个 state 上提供的局部监督。

```text
state_id: str
branch: str
target: tuple[float, ...]
signal_type: str
metadata: dict
```

对不同分支，`target` 的语义不同：

- AR：teacher token logprobs。
- diffusion：teacher noise prediction / score target。
- flow：teacher velocity field target。
- VLA：teacher action chunk / trajectory hint。

### `TeacherGuidedObjectiveConfig`

控制 clipping 和诊断。

```text
name: str
per_element_clip: float | None
branch: str
```

### `compute_teacher_guided_objective`

通用向量型 teacher-guided objective。

```text
student_state.payload 与 teacher_guidance.target 做逐元素 gap
raw_objective = mean(abs(gap))
objective = mean(clipped(abs(gap)))
```

这个函数不声称覆盖所有 OPD 形式。它是通用接口的第一版 reference objective，适合 AR sampled-token gap、diffusion denoising target、flow velocity target 等向量监督。

## AR 适配

新增两个 helper：

```text
rollout_batch_to_on_policy_states(batch, student_scores)
teacher_signals_to_guidance(signals)
```

它们把现有 AR OPD 路径转换到通用 OPD 抽象：

```text
RolloutBatch + student token logprobs
  -> OnPolicyState(branch="ar", payload=student_logprobs)

TeacherSignal
  -> TeacherGuidance(branch="ar", target=teacher_logprobs)
```

这样现有 `opd.py` 可以暂时保留，同时新接口能复用同样的数据。

## Diffusion / Flow Toy 验证

第一版只做 dependency-light toy：

```text
OnPolicyState(branch="diffusion", payload=(student_noise_pred...))
TeacherGuidance(branch="diffusion", target=(teacher_noise_pred...))
```

以及：

```text
OnPolicyState(branch="flow", payload=(student_velocity...))
TeacherGuidance(branch="flow", target=(teacher_velocity...))
```

目标是证明 MindRL 的 OPD 抽象不是 AR-only。真实 diffusion / flow 接入后，需要把 payload/target 换成模型真实暴露的 denoising score、velocity、path surrogate 或 action chunk。

## 诊断指标

通用 objective 输出：

- `raw_objective`
- `objective`
- `elements`
- `clipped_elements`
- `clipped_ratio`

这些指标用于判断 teacher signal 是否被少数高 gap 元素主导。AR 中这对应高 KL style token；diffusion/flow 中则对应异常 denoising target、velocity drift 或局部 surrogate spike。

## 设计边界

这个抽象是“广义 OPD 的数据和 objective 接口”，不是完整训练 runtime。真实训练仍由 branch-specific trainer 负责：

- AR：PEFT / TRL / verl / OpenRLHF。
- diffusion：DDPO / diffusers trainer / custom denoising trainer。
- flow：flow matching trainer。
- VLA：behavior cloning / RL policy trainer。

MindRL 的责任是把 reward 或 teacher signal 翻译成分支原生 update object，并暴露风险诊断。

## 实施计划

1. 新增 `src/mindrl/generalized_opd.py`。
2. 新增 `tests/test_generalized_opd.py`。
3. 先写测试覆盖：
   - AR batch/signals 能映射到通用 state/guidance。
   - 通用 objective 能计算 raw/clipped/clipped_ratio。
   - diffusion toy vector objective 可用。
   - flow toy vector objective 可用。
4. 实现最小代码。
5. 跑相关测试和完整测试。
