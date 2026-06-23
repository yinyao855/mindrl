# Quickstart: AR GRPO Smoke

This smoke path validates the AR LLM side without downloading a model.

```bash
uv run python examples/run_mvp_smoke.py --output-dir outputs/mvp_smoke
```

The AR report uses built-in math/code samples, exact-match rewards, and a
GRPO-style group-relative objective. It is deliberately small so researchers can
inspect the reward, advantage, and objective values before replacing the mock
rollouts with a real Qwen/Llama rollout engine.

Key objects:

- `mindrl.core.RolloutBatch`
- `mindrl.ar_training.exact_match_reward`
- `mindrl.ar_training.compute_grpo_objective`
- `mindrl.core.TrainReport`

Expected artifacts:

- `outputs/mvp_smoke/mvp_smoke_reports.jsonl`
- `outputs/mvp_smoke/mvp_smoke_report.md`
