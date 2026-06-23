# Custom Rewards

Rewards in `mindrl` return `RewardOutput`, keyed by rollout sample id.

```python
from mindrl.core import RewardOutput, RolloutBatch


def length_penalty_reward(batch: RolloutBatch) -> RewardOutput:
    rewards = {
        sample.sample_id: 1.0 / max(1, len(sample.response.split()))
        for sample in batch.samples
    }
    return RewardOutput(sample_rewards=rewards)
```

Guidelines:

- Keep reward keys identical to `RolloutBatch.sample_ids`.
- Store diagnostic details in `reward_metadata` when useful.
- Use verifiable rewards first for AR smoke runs.
- For diffusion, start with cheap deterministic proxies before learned reward
  models, so reward hacking is easy to inspect.
