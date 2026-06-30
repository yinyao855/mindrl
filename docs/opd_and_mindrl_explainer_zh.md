# OPD 与 MindRL 接口层说明

## 1. Teacher 是怎么来的

OPD 里的 teacher 不一定只有一种来源。常见有几类。

第一类是更强模型。比如 student 是 7B，teacher 是 32B/70B，teacher 在 student 生成的 prefix 上给 token logprob 或 logits。

第二类是经过 SFT/RL 后的专家模型。比如一个 math expert、code expert、tool-use expert。它不一定比 student 大很多，但在某个领域更强。

第三类是同一个模型的 privileged 版本，也就是 OPSD。teacher 和 student 可能参数相同，但 teacher 输入里多了参考答案、ground truth reasoning、测试结果等额外信息。例如：

```text
student 输入：
Question: 2+2?

teacher 输入：
Correct answer: 4
Question: 2+2?
Student response prefix: ...
```

这样 teacher 的 logprob 就带有“知道答案”的局部指导。

第四类是 reward model / verifier / process reward model。它不一定输出 token logits，而是输出状态分数、步骤分数或最终 reward。

所以 teacher 本质上不是“一个固定训练方法训练出来的东西”，而是：

```text
能在 student 访问的状态上提供更有用监督信号的模型或函数。
```

## 2. 可更新对象是什么

“可更新对象”指的是：reward 或 teacher signal 最终要通过什么数学量进入梯度更新。

对 AR 语言模型，最自然的可更新对象是 token logprob：

```text
log pi_theta(y_t | x, y_<t)
```

因为 AR 模型每一步都明确给出下一个 token 的概率。所以 PPO/GRPO/OPD 可以比较自然地写成 token 或 sequence 级目标。

例如 RL / GRPO 里常见对象是 ratio：

```text
r_t(theta) = pi_theta(y_t | s_t) / pi_ref(y_t | s_t)
```

或者 log-ratio：

```text
log pi_theta(y_t | s_t) - log pi_ref(y_t | s_t)
```

OPD 里常见对象是 teacher-student KL：

```text
D_KL(pi_T(. | s_t) || pi_theta(. | s_t))
```

或 reverse-KL / sampled-token 形式：

```text
log pi_theta(y_t | s_t) - log pi_T(y_t | s_t)
```

对 diffusion / flow，问题更复杂。它生成的不是离散 token，而是连续 latent、action、image trajectory。你不一定能便宜地算出：

```text
log pi_theta(a | s)
```

或者：

```text
pi_theta(a | s) / pi_ref(a | s)
```

所以不能直接把 token-style PPO/GRPO 接上去。可能只能用：

```text
flow matching loss
denoising score
surrogate reward
anchor distance
drift penalty
smoothness penalty
```

对 parallel discrete / dLLM，它一次更新多个 token 或 block。问题是 block 内 token 可能强依赖。如果把每个位置当独立 token 更新，就可能错配 joint distribution。它的可更新对象可能应该是：

```text
block-level score
dependence-aware block objective
nCTC-controlled granularity
```

所以“可更新对象”就是：当前 policy branch 实际暴露出来、能被梯度安全作用的概率量、score、surrogate 或结构约束项。

## 3. 为什么不能把 reward 直接接到所有 token / branch

假设有一个最终 reward：

```text
R(y) = 1  如果答案对
R(y) = 0  如果答案错
```

最天真的做法是：答案对了，就把整条轨迹所有 token 都强化；答案错了，就都压低。

但这有几个问题。

### Token credit 不均匀

一条推理里，有些 token 是关键数学步骤，有些只是风格词：

```text
"wait", "therefore", "let's see"
```

如果 teacher-student KL 最大的是这些风格 token，而不是关键数学 token，直接更新会让模型过度学习风格，甚至 collapse。

这就是博客里提到的：OPSD 中 style / pivot tokens 的 KL 可能比 math tokens 更高，所以需要 per-token clipping。

### Branch 的概率接口不同

AR token 可以算 logprob；flow/diffusion action chunk 可能没有便宜 exact density；dLLM block 有 factorization barrier。你不能假设所有分支都支持：

```text
reward -> token ratio -> PPO/GRPO update
```

### Signal 有 bias / variance / drift

teacher signal 不是 ground truth。它可能偏向 teacher 的风格、偏向某些高频模式，或者在 student 偏离 teacher 常见 prefix 后变得不可靠。

所以要看：

- bias：teacher signal 是否真的反映任务质量，还是反映风格。
- variance：token-level signal 是否波动很大。
- drift：更新后 policy 是否偏离原来的可行区域。
- clip risk：是否少数 token gap 过大，主导了 update。

MindRL 想做的就是：reward 进入 update 前，先问这个分支“你能安全暴露什么更新对象？这个 signal 风险多大？”

## 4. OPD 具体在干什么

设 student policy 是：

```text
pi_theta
```

teacher 是：

```text
pi_T
```

prompt 是 `x`。OPD 先让 student 生成：

```text
y ~ pi_theta(. | x)
```

对每个 prefix：

```text
s_t = (x, y_<t)
```

teacher 给出：

```text
pi_T(. | s_t)
```

一种 supervised OPD 目标可以写成：

```text
L_OPD(theta)
= E_{x, y ~ pi_theta} [ sum_t D_KL(pi_T(. | s_t) || pi_theta(. | s_t)) ]
```

这表示：训练状态来自 student，但监督分布来自 teacher。

如果只看 sampled token，也可以近似成：

```text
L ~= - sum_t w_t log pi_theta(y_t | s_t)
```

其中 `w_t` 来自 teacher-student logprob 差、KL 或 teacher confidence。

Policy-gradient OPD 可以写成更像 RL 的形式：

```text
grad_theta J(theta)
~= E_{y ~ pi_theta} [ sum_t A_t^teacher grad_theta log pi_theta(y_t | s_t) ]
```

其中 teacher-derived advantage 可能类似：

```text
A_t^teacher = log pi_T(y_t | s_t) - log pi_theta(y_t | s_t)
```

直觉是：

- teacher 比 student 更喜欢这个 token，就提高它概率。
- teacher 比 student 更不喜欢这个 token，就降低它概率。

但这个 `A_t^teacher` 有 bias，因为 teacher 喜欢的不一定是任务关键 token，所以需要 clipping、entropy diagnostic、support filtering 等稳定化机制。

## 5. SFT / RL / OPD 的公式对比

SFT：

```text
L_SFT(theta)
= - E_{(x,y*) ~ D} [ sum_t log pi_theta(y*_t | x, y*_<t) ]
```

特点：

```text
状态来自固定数据 D
信号是 dense token label
```

RL：

```text
J_RL(theta)
= E_{y ~ pi_theta(.|x)} [ R(x,y) ]
```

policy gradient：

```text
grad_theta J
~= E [ sum_t A(x,y) grad_theta log pi_theta(y_t | s_t) ]
```

特点：

```text
状态来自 student
信号通常是 sparse reward
```

OPD：

```text
L_OPD(theta)
= E_{y ~ pi_theta(.|x)} [ sum_t D_KL(pi_T(.|s_t) || pi_theta(.|s_t)) ]
```

特点：

```text
状态来自 student
信号来自 teacher，且通常是 dense token-level
```

所以 OPD 可以说是 SFT 和 RL 的中间体：

```text
SFT: off-policy states + dense supervision
RL:  on-policy states  + sparse reward
OPD: on-policy states  + dense teacher supervision
```

## 6. MindRL 的核心主张

MindRL 比 OPD 更一般。

OPD 主要讨论 AR LLM 上：

```text
student rollout + teacher token distribution
```

MindRL 关心的是多分支生成策略：

```text
AR token branch
parallel discrete branch
diffusion / flow branch
VLA action branch
agent tool/action branch
```

它的核心主张是：

```text
Reward is not a universal interface.
```

更完整地说：

```text
一个 scalar reward 不能自动成为所有生成策略分支的合法 update。
它必须先通过该分支实际暴露的 probability object / score object / surrogate object。
```

MindRL 的抽象是：

```text
PolicySpec -> BarrierProfile -> AdapterDecision
```

也就是：

1. 先看分支是什么。
2. 再测这个分支的接口障碍。
3. 再决定 reward / teacher signal 应该怎么进入 loss。

### AR branch

如果是 AR，且 exact logprob 可用：

```text
adapter = exact_ratio
loss ~= - A * log pi_theta(y_t | s_t)
```

### parallel discrete branch

如果是 dLLM/block generation，且 nCTC 高：

```text
adapter = dependence_aware_block
granularity ↓
structure_cost_weight ↑
```

意思是 block 内依赖强，就不要粗暴并行更新大 block。

### diffusion / flow branch

如果是 flow/diffusion，且 surrogate variance 或 drift 高：

```text
adapter = anchored_flow_surrogate
anchor_strength ↑
clip_range ↓
branch_weight ↓
```

意思是 surrogate 不可信或 policy 漂移大，就降低这条 branch 的更新强度，加强 anchor。

## 7. MindRL 是不是 OPD 和 RL/SFT 的中间体

不是完全一样。

OPD 可以看成：

```text
AR LLM 上的一种 on-policy + teacher-guided 方法
```

它确实位于 SFT 和 RL 之间。

MindRL 是更上层的 interface controller。它可以包含 OPD，但不等于 OPD。

更准确地说：

```text
OPD 是一种 branch-native update 方法。
MindRL 是决定不同 branch 应该使用哪种 update 方法的控制层。
```

在 AR 分支上，MindRL 可能选择：

```text
GRPO
OPD
SFT-style LoRA
REINFORCE baseline
```

在 diffusion 分支上，MindRL 可能选择：

```text
DDPO
anchored flow surrogate
reward reranking
smoothness penalty
```

在 dLLM 分支上，MindRL 可能选择：

```text
adaptive block size
dependence-aware objective
selective serialization
```

所以：

```text
OPD 是一个算法。
MindRL 是一个“reward/teacher signal 如何接入不同生成分支”的框架。
```

## 8. 用一个统一公式看 MindRL

可以把多分支 policy 写成：

```text
pi_theta(y | x) = product_b pi_theta,b(y_b | s_b)
```

其中 `b` 是分支：

```text
b in {AR, dLLM, diffusion, flow, tool, action}
```

普通做法可能是：

```text
L = - R(y) sum_b log pi_theta,b(y_b | s_b)
```

这等于把同一个 reward 直接砸到所有 branch 上。

MindRL 会改成：

```text
L_MindRL
= sum_b w_b * A_b(R, T, s_b) * U_b(theta; y_b, s_b)
   + lambda_b * C_b(theta; s_b)
```

其中：

- `w_b` 是 branch weight。
- `A_b` 是该分支的 reward / teacher-derived signal。
- `U_b` 是该分支真正可更新对象。
- `C_b` 是结构约束或 anchor penalty。
- `lambda_b` 是控制强度。

不同分支的 `U_b` 不一样。

AR：

```text
U_AR = - log pi_theta(y_t | s_t)
```

OPD：

```text
U_OPD = D_KL(pi_T(.|s_t) || pi_theta(.|s_t))
```

dLLM：

```text
U_dLLM = block-level dependence-aware loss
C_dLLM = nCTC penalty / granularity cost
```

flow/diffusion：

```text
U_flow = surrogate score / denoising objective
C_flow = anchor distance + drift + smoothness
```

controller 决定：

```text
(w_b, lambda_b, clip_b, granularity_b)
= f(PolicySpec_b, BarrierProfile_b)
```

这就是 MindRL 的核心。

## 9. MindRL 能不能把 OPD 抽象为通用方法，用在 diffusion 等模型上

可以，但要非常小心：不能把 AR token OPD 原样搬到 diffusion / flow 上。

更准确的说法是：MindRL 可以把 OPD 抽象成一种更一般的 **teacher-guided on-policy update**：

```text
student 自己采样状态
teacher 在 student 状态上给局部监督
student 用 branch-native update object 学 teacher
```

这个抽象对不同分支都成立：

```text
AR:
  state = prompt + prefix
  teacher signal = token logits / logprobs
  update object = token KL / logprob ratio

diffusion:
  state = noisy latent + timestep + condition
  teacher signal = teacher denoising prediction / score / clean latent estimate
  update object = denoising loss / score matching loss / path surrogate

flow:
  state = latent path point + time + condition
  teacher signal = teacher velocity field / flow direction / energy guidance
  update object = flow matching surrogate + anchor / drift penalty

VLA action:
  state = observation + language context + action history
  teacher signal = action chunk / trajectory hint / world-model correction
  update object = behavior cloning term + smoothness / contact / safety cost
```

也就是说，OPD 的“on-policy + dense teacher signal”思想可以泛化；但具体 loss 必须换成该分支原生的可更新对象。

例如 diffusion 上的类 OPD 可能写成：

```text
z_t ~ student diffusion path
teacher predicts epsilon_T(z_t, t, c)
student predicts epsilon_theta(z_t, t, c)

L_diffusion_OPD
= E_{z_t sampled from student path} [
     || epsilon_theta(z_t,t,c) - epsilon_T(z_t,t,c) ||^2
   + alpha * anchor_distance
   + beta  * smoothness_penalty
]
```

这和 AR OPD 的形式相似：

```text
student 采样自己的状态
teacher 在这些状态上提供 dense target
student 匹配 teacher
```

但它不是：

```text
D_KL(pi_T(token | prefix) || pi_theta(token | prefix))
```

因为 diffusion 分支未必暴露 token probability。

所以 MindRL 可以做“广义 OPD”，但不是把 AR OPD 公式复制到 diffusion。

## 10. 当前 MindRL 框架支持哪些模型

当前支持分成三层：真实已跑通、代码接口支持、研究原型支持。

### 真实已跑通的 Hugging Face causal LM

这些模型已经在真实 smoke 中跑通过：

- `sshleifer/tiny-gpt2`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct` 的两卡 load/inference

其中：

- Qwen3-0.6B：AR smoke、OPD scoring、PEFT SFT、PEFT OPD 都跑通过。
- Qwen2.5-7B：AR smoke、PEFT SFT 跑通过。
- Qwen2.5-14B：两卡 load-only + short inference 跑通过；PEFT 尚未支持。

### 理论上支持的 AR causal LM

只要模型能通过：

```text
AutoModelForCausalLM
AutoTokenizer
```

加载，并支持 teacher-forced logprob 计算，就可以接入：

- `HFCausalLMGroupPolicy`
- `HFCausalLMTeacherSignalAdapter`
- `run_real_ar_smoke.py`
- `run_real_opd_smoke.py`
- `run_peft_sft_smoke.py`

大模型需要显式设置：

```text
--dtype fp16
```

或后续增加 `device_map` 支持。

### diffusion / flow / dLLM 原型

当前 MindRL 有这些接口原型：

- diffusion DDPO smoke
- mock diffusion rollout adapter
- flow surrogate / drift / anchor diagnostics
- parallel discrete nCTC proxy
- dLLM decoding utilities
- VLA interface prototype
- agentic barrier metrics

但这些多数是 dependency-light prototype，还没有像 Qwen causal LM 那样完成大模型真实训练路径。

### 当前不支持或暂未跑通的模型

- Dream 7B：本地 snapshot 缺 `modeling_dream.py`，模型加载失败。
- LLaDA 8B：模型代码和当前 `transformers 5.12.1` 不兼容，加载后报 `all_tied_weights_keys` 缺失。
- Qwen2.5-14B PEFT：需要 multi-GPU/device-map-aware PEFT、量化或 offload，目前只跑了 load/inference。

## 11. 最简总结

OPD 的主张：

```text
不要只在 teacher 轨迹上学。
让 student 自己 rollout，再让 teacher 在 student 状态上指导。
```

MindRL 的主张：

```text
不要假设 reward / teacher signal 可以直接更新所有 token 和 branch。
先识别每个 branch 暴露的可更新对象，
再根据接口障碍决定 adapter、clip、anchor、granularity 和 branch weight。
```

所以 MindRL 不是简单的 OPD 变体。它更像是一个总控层，OPD 是它在 AR teacher-guided 场景下可以调用的一种 branch-native update。MindRL 也可以把 OPD 的思想推广成“teacher-guided on-policy update”，但推广到 diffusion / flow / VLA 时，必须换成对应分支原生的 score、surrogate、anchor 或结构约束。
