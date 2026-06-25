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
  - grouped rollout + reward adapter for GRPO
  - exact-match verifiable reward
  - GRPO-style group-relative objective
  - OPD-style student rollout + teacher signal adapter
  - OPD per-token clipping and entropy diagnostics
  - REINFORCE++/OPO-like batch-baseline objective
- AR trainer planning:
  - Qwen 0.5B/1.5B/7B/13B LoRA and QLoRA presets
  - effective batch size and rough VRAM estimates
- Diffusion objectives:
  - diffusers-style pipeline adapter protocol
  - DDPO-style denoising trajectory objective
  - compressibility reward
  - CLIP-like prompt/caption alignment proxy
  - image-grid manifest for report review
  - controller-backed diffusion run summary
- External framework exports:
  - TRL GRPO config dictionary
  - verl OPD config dictionary
  - OpenRLHF GRPO command arguments
- Controller ablation:
  - uniform reward
  - score routing
  - barrier-gated controller
- Smoke benchmark/report path:
  - AR GRPO smoke
  - AR OPD smoke
  - AR LoRA plan smoke
  - diffusion DDPO smoke
  - controller ablation smoke
  - JSONL + Markdown report output

## Run the MVP Smoke

```bash
uv run python examples/run_mvp_smoke.py
uv run mindrl
```

Outputs are written to `outputs/mvp_smoke/`.

## Run Real Model Smokes

These commands use local Hugging Face snapshots when available:

```bash
uv run python examples/run_real_ar_smoke.py --model <local-causal-lm-snapshot>
uv run python examples/run_real_opd_smoke.py --student-model <local-causal-lm-snapshot>
uv run python examples/run_peft_sft_smoke.py --model <local-causal-lm-snapshot>
```

## Test

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

## Docs

- `docs/quickstart_ar_grpo.md`
- `docs/quickstart_ar_opd.md`
- `docs/quickstart_ar_lora.md`
- `docs/quickstart_diffusion_ddpo.md`
- `docs/custom_reward.md`
- `docs/troubleshooting.md`
- `docs/framework_integration_decision.md`
- `docs/mvp_sync_2026_07_25.md`

## Repository Layout

```text
src/mindrl/
  core.py                    # rollout, reward, teacher signal, report objects
  ar_training.py             # GRPO, OPD, exact reward, baseline objective
  grpo.py                    # grouped rollout/reward loop for GRPO
  opd.py                     # on-policy distillation rollout/teacher loop
  ar_trainer.py              # LoRA/QLoRA training plan presets
  external_adapters.py       # TRL/verl/OpenRLHF config exporters
  diffusion_adapter.py       # diffusers-style rollout adapter and image manifest
  diffusion_training.py      # DDPO-style trajectory objective and rewards
  interface_controller.py    # pluggable MindRL controller primitives
  mvp_benchmarks.py          # dependency-light AR + diffusion smoke reports
tests/
examples/
docs/
```

## Research Position

The MVP is not another distributed RL runtime. UniRL, verl, TRL, OpenRLHF, and
DDPO implementations already cover large parts of rollout/training execution.
MindRL focuses on the transparent interface layer that turns rewards and teacher
signals into branch-native update objects for AR LLM and diffusion policies.
