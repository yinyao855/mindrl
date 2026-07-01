import unittest

from mindrl.core import RewardOutput, RolloutBatch, RolloutSample


class PPOStyleObjectiveTest(unittest.TestCase):
    def test_ppo_style_objective_uses_group_relative_advantages_and_clipping(self):
        from mindrl.ppo_style import compute_ppo_style_objective

        batch = RolloutBatch(
            samples=(
                RolloutSample("s1", "p", "4", "ar", {"prompt_id": "p"}),
                RolloutSample("s2", "p", "5", "ar", {"prompt_id": "p"}),
            )
        )
        rewards = RewardOutput({"s1": 1.0, "s2": 0.0})

        objective = compute_ppo_style_objective(
            batch=batch,
            rewards=rewards,
            logprob_ratios={"s1": 1.5, "s2": 0.5},
            kl_by_sample={"s1": 0.2, "s2": 0.1},
            clip_range=0.2,
            kl_weight=0.1,
        )

        self.assertLess(objective.objective, 0.0)
        self.assertAlmostEqual(objective.sample_weights["s1"], 0.5)
        self.assertAlmostEqual(objective.sample_weights["s2"], -0.5)
        self.assertIn("clipped_policy_term", objective.diagnostics)
        self.assertAlmostEqual(objective.diagnostics["kl"], 0.15)

    def test_ppo_style_objective_is_zero_when_rewards_are_equal(self):
        from mindrl.ppo_style import compute_ppo_style_objective

        batch = RolloutBatch(
            samples=(
                RolloutSample("s1", "p", "4", "ar", {"prompt_id": "p"}),
                RolloutSample("s2", "p", "4", "ar", {"prompt_id": "p"}),
            )
        )
        rewards = RewardOutput({"s1": 1.0, "s2": 1.0})

        objective = compute_ppo_style_objective(
            batch=batch,
            rewards=rewards,
            logprob_ratios={"s1": 1.2, "s2": 0.8},
        )

        self.assertAlmostEqual(objective.diagnostics["policy_term"], 0.0)
        self.assertAlmostEqual(objective.objective, -0.0)


if __name__ == "__main__":
    unittest.main()
