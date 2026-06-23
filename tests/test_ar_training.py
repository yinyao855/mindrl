import unittest

from mindrl.ar_training import (
    compute_grpo_objective,
    compute_opd_objective,
    compute_reinforce_baseline_objective,
    exact_match_reward,
)
from mindrl.core import RewardOutput, RolloutBatch, RolloutSample, TeacherSignal


class ARTrainingTest(unittest.TestCase):
    def test_exact_match_reward_scores_normalized_answers(self):
        batch = RolloutBatch(
            samples=(
                RolloutSample("s1", "2+2", " 4 ", "ar", {"answer": "4"}),
                RolloutSample("s2", "2+3", "six", "ar", {"answer": "5"}),
            )
        )

        rewards = exact_match_reward(batch)

        self.assertEqual(rewards.sample_rewards, {"s1": 1.0, "s2": 0.0})

    def test_grpo_objective_uses_group_relative_advantages(self):
        batch = RolloutBatch(
            samples=(
                RolloutSample("p1-a", "p1", "good", "ar", {"prompt_id": "p1"}),
                RolloutSample("p1-b", "p1", "bad", "ar", {"prompt_id": "p1"}),
                RolloutSample("p2-a", "p2", "ok", "ar", {"prompt_id": "p2"}),
            )
        )
        rewards = RewardOutput({"p1-a": 1.0, "p1-b": 0.0, "p2-a": 0.5})

        objective = compute_grpo_objective(
            batch,
            rewards,
            logprob_ratios={"p1-a": 1.1, "p1-b": 0.9, "p2-a": 1.0},
            kl_by_sample={"p1-a": 0.1, "p1-b": 0.2, "p2-a": 0.0},
            kl_weight=0.1,
        )

        self.assertAlmostEqual(objective.sample_weights["p1-a"], 0.5)
        self.assertAlmostEqual(objective.sample_weights["p1-b"], -0.5)
        self.assertAlmostEqual(objective.sample_weights["p2-a"], 0.0)
        self.assertLess(objective.objective, 0.0)
        self.assertIn("reward_mean", objective.diagnostics)

    def test_opd_objective_matches_student_rollouts_to_teacher_signal(self):
        objective = compute_opd_objective(
            student_logprobs={
                "s1": (-0.4, -0.6),
                "s2": (-0.2,),
            },
            teacher_signals=(
                TeacherSignal("s1", (-0.1, -0.3)),
                TeacherSignal("s2", (-0.5,)),
            ),
        )

        self.assertAlmostEqual(objective.objective, 0.3)
        self.assertEqual(objective.sample_weights, {"s1": 0.3, "s2": 0.3})
        self.assertAlmostEqual(objective.diagnostics["tokens"], 3.0)

    def test_reinforce_baseline_centers_rewards_by_batch_mean(self):
        rewards = RewardOutput({"s1": 1.0, "s2": 0.0, "s3": 0.5})

        objective = compute_reinforce_baseline_objective(
            rewards,
            logprobs={"s1": -0.2, "s2": -0.4, "s3": -0.1},
        )

        self.assertAlmostEqual(objective.sample_weights["s1"], 0.5)
        self.assertAlmostEqual(objective.sample_weights["s2"], -0.5)
        self.assertAlmostEqual(objective.sample_weights["s3"], 0.0)
        self.assertAlmostEqual(objective.diagnostics["baseline"], 0.5)


if __name__ == "__main__":
    unittest.main()
