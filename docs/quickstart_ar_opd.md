# Quickstart: AR OPD Step

On-policy distillation (OPD) trains on states visited by the student policy.
The first MVP exposes the student rollout, teacher signal adapter, token-level
clipping, and report path separately from a heavyweight trainer:

```python
from mindrl.opd import (
    MappingTeacherSignalAdapter,
    MockARPolicy,
    OPDConfig,
    run_opd_step,
)

student = MockARPolicy(
    responses={"math": "wait add four"},
    token_logprobs={"opd-0": (-1.3, -0.4, -0.3)},
)
teacher = MappingTeacherSignalAdapter(
    token_logprobs={"opd-0": (-0.1, -0.2, -0.2)},
    entropies={"opd-0": (1.5, 0.4, 0.3)},
)
result = run_opd_step(("math",), student, teacher, OPDConfig(per_token_clip=0.25))
```

Interpretation:

- The student creates the rollout states, preserving the on-policy property.
- The teacher only scores those student states with dense token logprobs.
- `per_token_clip` limits high-KL style/pivot tokens from dominating updates.
- The report tracks `raw_objective`, `clipped_tokens`, and
  `mean_teacher_entropy`.

Real-model integration should add a rollout engine and teacher-logprob adapter,
not change the objective API.
