# mindrl

`mindrl` is a transparent RL infrastructure pilot for AR LLM and diffusion
research. It grew out of the MINDRL reproduction code, but the new repository is
organized around framework primitives: rollout batches, rewards, teacher
signals, branch-native objectives, reports, and a lightweight
`PolicySpec -> BarrierProfile -> AdapterDecision` interface controller.

The first MVP intentionally avoids rebuilding a verl/OpenRLHF-style distributed
runtime. It focuses on code that researchers can read, test, and modify before
plugging into TRL, verl, OpenRLHF, or a diffusion training stack.

## MVP Features

- Core framework objects:
  - `RolloutSample` / `RolloutBatch`
  - `RewardOutput`
  - `TeacherSignal`
  - `AlgorithmConfig` / `ObjectiveOutput` / `TrainReport`
- AR LLM objectives:
  - exact-match verifiable reward
  - GRPO-style group-relative objective
  - OPD-style token-level teacher/student matching
  - REINFORCE++/OPO-like batch-baseline objective
- Diffusion objectives:
  - DDPO-style denoising trajectory objective
  - compressibility reward
  - CLIP-like prompt/caption alignment proxy
  - controller-backed diffusion run summary
- Smoke benchmark/report path:
  - AR GRPO smoke
  - diffusion DDPO smoke
  - JSONL + Markdown report output

## Run the MVP Smoke

```bash
uv run python examples/run_mvp_smoke.py
uv run mindrl
```

Outputs are written to `outputs/mvp_smoke/`.

## Run

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
uv run python examples/offline_random_ar_proxy_nctc.py
```

Chinese reproduction notes:

- `docs/reproduction_runbook_zh.md`
- `docs/mindrl_80_reproduction_report_zh.md`

CPU-only task-level pilot without any model download:

```bash
uv run python examples/run_task_nctc_proxy.py --sample preset --max-examples 3 --device cpu --mock-scorer
bash examples/run_cpu_pilot.sh
```

The `run_cpu_pilot.sh` and `run_gpu_pilot.sh` helpers use `uv run python` when
`uv` is installed, otherwise they fall back to `PYTHONPATH=src python3`.

When Hugging Face cache/downloads are available, run the same task-level proxy
with a real causal-LM scorer:

```bash
uv run python examples/hf_ar_proxy_nctc.py --model sshleifer/tiny-gpt2
uv run python examples/hf_ar_proxy_nctc.py --model Qwen/Qwen3-0.6B
uv run python examples/run_task_nctc_proxy.py --model sshleifer/tiny-gpt2 --device cpu --sample preset --max-examples 3 --local-files-only
MODEL=Qwen/Qwen3-0.6B MAX_EXAMPLES=20 bash examples/run_gpu_pilot.sh
```

## Repository Layout

```text
src/mindrl/
  discrete_interface.py      # nCTC and adaptive block utilities
  synthetic_barriers.py      # finite-distribution CTC checks
  ar_proxy_nctc.py           # causal-LM dependency-gap proxy
  benchmark_tasks.py         # preset/JSONL task samples
  hf_scorer.py               # Hugging Face teacher-forcing scorer
  interface_controller.py    # pluggable MINDRL controller primitives
  agentic_barriers.py        # toy agent trace barrier metrics
tests/
examples/
docs/
```

## Research Position

The reproduction target is intentionally narrow: the discrete language-side
nCTC/adaptive-block mechanism is the easiest part to verify without the full
8 x H100 paired AR-to-dLLM setup from the paper.

The task-level script reports an AR-scorer proxy:

```text
joint = log p_AR(completion block | prompt)
marginal = sum_i log p_AR(token_i | prompt)
```

This is useful for checking dependence trends and pipeline viability, but it is
not the paper's paired AR-to-dLLM nCTC protocol or fixed-vs-adaptive dLLM
decoding result.

For new work, the recommended direction is not another full RL framework. UniRL,
verl, TRL, and Agent Lightning already cover much of the rollout/training
runtime. The better research object is a plug-in reward-to-update interface
controller that decides how each branch receives reward updates.
