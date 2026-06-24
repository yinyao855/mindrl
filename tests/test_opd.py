import unittest

from mindrl.opd import (
    MappingTeacherSignalAdapter,
    MockARPolicy,
    OPDConfig,
    compute_clipped_opd_objective,
    run_opd_step,
)
from mindrl.core import TeacherSignal


class OPDLoopTest(unittest.TestCase):
    def test_mock_policy_rolls_out_on_policy_samples_and_scores_them(self):
        policy = MockARPolicy(
            responses={"p1": "wait add four", "p2": "answer five"},
            token_logprobs={
                "opd-0": (-1.0, -0.3, -0.2),
                "opd-1": (-0.5, -0.4),
            },
        )

        batch = policy.rollout(("p1", "p2"))
        scores = policy.score(batch)

        self.assertEqual(batch.sample_ids, ("opd-0", "opd-1"))
        self.assertEqual(batch.samples[0].response, "wait add four")
        self.assertEqual(scores["opd-0"], (-1.0, -0.3, -0.2))

    def test_teacher_adapter_matches_student_rollout_states(self):
        policy = MockARPolicy(
            responses={"p1": "wait add four"},
            token_logprobs={"opd-0": (-1.0, -0.3, -0.2)},
        )
        teacher = MappingTeacherSignalAdapter(
            token_logprobs={"opd-0": (-0.1, -0.2, -0.2)},
            topk_tokens={"opd-0": (("ok",), ("add",), ("four",))},
            entropies={"opd-0": (1.5, 0.4, 0.3)},
        )

        batch = policy.rollout(("p1",))
        signals = teacher.signals_for(batch)

        self.assertEqual(signals[0].sample_id, "opd-0")
        self.assertEqual(signals[0].token_count, 3)
        self.assertEqual(signals[0].topk_tokens[1], ("add",))

    def test_clipped_opd_objective_limits_high_kl_style_tokens(self):
        objective = compute_clipped_opd_objective(
            student_logprobs={"s1": (-1.3, -0.4, -0.3)},
            teacher_signals=(TeacherSignal("s1", (-0.1, -0.2, -0.2)),),
            config=OPDConfig(per_token_clip=0.25),
            teacher_entropies={"s1": (1.5, 0.4, 0.3)},
        )

        self.assertAlmostEqual(objective.objective, 0.183333333333)
        self.assertAlmostEqual(objective.diagnostics["raw_objective"], 0.5)
        self.assertAlmostEqual(objective.diagnostics["clipped_tokens"], 1.0)
        self.assertAlmostEqual(objective.diagnostics["mean_teacher_entropy"], 0.733333333333)

    def test_run_opd_step_returns_reportable_objective(self):
        policy = MockARPolicy(
            responses={"p1": "wait add four"},
            token_logprobs={"opd-0": (-1.3, -0.4, -0.3)},
        )
        teacher = MappingTeacherSignalAdapter(
            token_logprobs={"opd-0": (-0.1, -0.2, -0.2)},
            entropies={"opd-0": (1.5, 0.4, 0.3)},
        )

        result = run_opd_step(("p1",), policy, teacher, OPDConfig(per_token_clip=0.25))

        self.assertEqual(result.batch.sample_ids, ("opd-0",))
        self.assertEqual(result.report.algorithm.name, "opd")
        self.assertIn("clipped_tokens", result.report.metrics)


if __name__ == "__main__":
    unittest.main()
