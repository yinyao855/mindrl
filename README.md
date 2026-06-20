# mindrl-repo

`mindrl-repo` is a small uv project for reproducing the low-cost parts of
`Reward Is Not a Universal Interface for Generative Reinforcement Learning` and
testing a possible extension toward interface-native agentic RL.

## What Is Implemented

- Discrete-side nCTC utilities:
  - distance-controlled block sampling
  - pair-normalized nCTC estimation from logprobs
  - inverse uncertainty adaptive block scheduling
- Synthetic CTC barrier checks for factorized parallel blocks.
- A lightweight `PolicySpec -> BarrierProfile -> AdapterDecision` controller.
- Toy agentic trace barrier metrics for tool/delegation/stop failures.

## Run

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python examples/minimal_discrete_repro.py
uv run python examples/offline_random_ar_proxy_nctc.py
```

When Hugging Face downloads are available:

```bash
uv run python examples/hf_ar_proxy_nctc.py --model sshleifer/tiny-gpt2
uv run python examples/hf_ar_proxy_nctc.py --model Qwen/Qwen3-0.6B
```

## Repository Layout

```text
src/mindrl_repo/
  discrete_interface.py      # nCTC and adaptive block utilities
  synthetic_barriers.py      # finite-distribution CTC checks
  ar_proxy_nctc.py           # causal-LM dependency-gap proxy
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

For new work, the recommended direction is not another full RL framework. UniRL,
verl, TRL, and Agent Lightning already cover much of the rollout/training
runtime. The better research object is a plug-in reward-to-update interface
controller that decides how each branch receives reward updates.
