# Quickstart: Diffusion DDPO Smoke

The first diffusion path is dependency-light. It treats a denoising trajectory as
step log probabilities plus an anchor distance, then computes a DDPO-style
score-function objective.

```python
from mindrl.diffusion_training import (
    DiffusionTrajectory,
    compressibility_reward,
    compute_ddpo_objective,
)

trajectories = (
    DiffusionTrajectory("img-a", "red cat", (-0.1, -0.2), anchor_distance=0.1),
    DiffusionTrajectory("img-b", "red cat", (-0.4, -0.3), anchor_distance=0.3),
)
rewards = compressibility_reward({"img-a": "aa", "img-b": "aaaaaaaa"})
objective = compute_ddpo_objective(trajectories, rewards, kl_anchor_weight=0.1)
```

Why this is separate from AR:

- AR LLMs expose token ratios directly.
- Diffusion policies expose denoising-step likelihood/surrogate quantities.
- `MindRLController` can still consume the resulting barrier profile, but the
  branch-native objective should remain diffusion-specific.

Future real-model adapters should populate `DiffusionTrajectory` from diffusers
or DDPO LoRA training code.
