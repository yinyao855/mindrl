# Quickstart: AR OPD Objective

On-policy distillation (OPD) trains on states visited by the student policy.
The first MVP exposes the objective separately from a heavyweight trainer:

```python
from mindrl.ar_training import compute_opd_objective
from mindrl.core import TeacherSignal

objective = compute_opd_objective(
    student_logprobs={"s1": (-0.4, -0.6)},
    teacher_signals=(TeacherSignal("s1", (-0.1, -0.3)),),
)
```

Interpretation:

- `student_logprobs` comes from the current student rollout.
- `TeacherSignal` stores dense teacher token log probabilities on the same
  rollout states.
- The MVP objective reports token-level mismatch and per-sample weights.

Real-model integration should add a rollout engine and teacher-logprob adapter,
not change the objective API.
