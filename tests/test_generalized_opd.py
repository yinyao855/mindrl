import unittest

from mindrl.core import RolloutBatch, RolloutSample, TeacherSignal


class GeneralizedOPDTest(unittest.TestCase):
    def test_ar_rollout_and_teacher_signals_convert_to_generic_states(self):
        from mindrl.generalized_opd import (
            rollout_batch_to_on_policy_states,
            teacher_signals_to_guidance,
        )

        batch = RolloutBatch(
            samples=(
                RolloutSample("s1", "p", "answer", "ar", {"prompt_id": "p"}),
            )
        )
        states = rollout_batch_to_on_policy_states(
            batch,
            student_scores={"s1": (-1.0, -0.4)},
        )
        guidance = teacher_signals_to_guidance(
            (TeacherSignal("s1", (-0.5, -0.2)),)
        )

        self.assertEqual(states[0].state_id, "s1")
        self.assertEqual(states[0].branch, "ar")
        self.assertEqual(states[0].payload, (-1.0, -0.4))
        self.assertEqual(states[0].metadata["prompt_id"], "p")
        self.assertEqual(guidance[0].target, (-0.5, -0.2))
        self.assertEqual(guidance[0].signal_type, "token_logprob")

    def test_teacher_guided_objective_reports_clipping(self):
        from mindrl.generalized_opd import (
            OnPolicyState,
            TeacherGuidance,
            TeacherGuidedObjectiveConfig,
            compute_teacher_guided_objective,
        )

        objective = compute_teacher_guided_objective(
            states=(
                OnPolicyState("s1", "ar", (-1.3, -0.4, -0.3)),
            ),
            guidance=(
                TeacherGuidance("s1", "ar", (-0.1, -0.2, -0.2), "token_logprob"),
            ),
            config=TeacherGuidedObjectiveConfig(
                name="generic-opd",
                branch="ar",
                per_element_clip=0.25,
            ),
        )

        self.assertAlmostEqual(objective.objective, 0.183333333333)
        self.assertAlmostEqual(objective.diagnostics["raw_objective"], 0.5)
        self.assertAlmostEqual(objective.diagnostics["clipped_ratio"], 1 / 3)
        self.assertEqual(objective.sample_weights["s1"], 0.183333333333)

    def test_diffusion_teacher_guided_vector_objective(self):
        from mindrl.generalized_opd import (
            OnPolicyState,
            TeacherGuidance,
            TeacherGuidedObjectiveConfig,
            compute_teacher_guided_objective,
        )

        objective = compute_teacher_guided_objective(
            states=(
                OnPolicyState("img-1", "diffusion", (0.2, 0.4, 0.6)),
            ),
            guidance=(
                TeacherGuidance("img-1", "diffusion", (0.1, 0.7, 0.5), "denoising_score"),
            ),
            config=TeacherGuidedObjectiveConfig(
                name="diffusion-teacher-guided",
                branch="diffusion",
                per_element_clip=None,
            ),
        )

        self.assertAlmostEqual(objective.objective, (0.1 + 0.3 + 0.1) / 3)
        self.assertEqual(objective.diagnostics["elements"], 3.0)

    def test_flow_teacher_guided_vector_objective_with_anchor_metadata(self):
        from mindrl.generalized_opd import (
            OnPolicyState,
            TeacherGuidance,
            TeacherGuidedObjectiveConfig,
            compute_teacher_guided_objective,
        )

        objective = compute_teacher_guided_objective(
            states=(
                OnPolicyState("flow-1", "flow", (1.0, 2.0), {"anchor_distance": 0.3}),
            ),
            guidance=(
                TeacherGuidance("flow-1", "flow", (0.5, 1.5), "velocity"),
            ),
            config=TeacherGuidedObjectiveConfig(
                name="flow-teacher-guided",
                branch="flow",
                per_element_clip=0.4,
            ),
        )

        self.assertAlmostEqual(objective.objective, 0.4)
        self.assertAlmostEqual(objective.diagnostics["raw_objective"], 0.5)
        self.assertAlmostEqual(objective.diagnostics["clipped_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
