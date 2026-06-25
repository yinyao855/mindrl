import unittest

from mindrl.grpo import (
    ExactAnswerRewardAdapter,
    GRPOConfig,
    MockGroupRolloutPolicy,
    NumericAnswerRewardAdapter,
    run_grpo_step,
)


class GRPORolloutTest(unittest.TestCase):
    def test_mock_group_policy_generates_grouped_rollouts(self):
        policy = MockGroupRolloutPolicy(
            completions={
                "math": ("4", "5"),
                "code": ("true", "false"),
            },
            logprob_ratios={
                "grpo-0-0": 1.1,
                "grpo-0-1": 0.9,
                "grpo-1-0": 1.0,
                "grpo-1-1": 1.0,
            },
        )

        batch = policy.rollout(("math", "code"), group_size=2)

        self.assertEqual(batch.sample_ids, ("grpo-0-0", "grpo-0-1", "grpo-1-0", "grpo-1-1"))
        self.assertEqual(batch.samples[0].metadata["prompt_id"], "math")
        self.assertEqual(batch.samples[2].metadata["group_index"], 0)

    def test_exact_answer_reward_adapter_scores_grouped_rollouts(self):
        policy = MockGroupRolloutPolicy(
            completions={"math": ("4", "5")},
            logprob_ratios={"grpo-0-0": 1.0, "grpo-0-1": 1.0},
        )
        batch = policy.rollout(("math",), group_size=2)
        reward = ExactAnswerRewardAdapter({"math": "4"}).score(batch)

        self.assertEqual(reward.sample_rewards, {"grpo-0-0": 1.0, "grpo-0-1": 0.0})

    def test_numeric_answer_reward_accepts_correct_answer_prefix(self):
        policy = MockGroupRolloutPolicy(
            completions={"math": ("4, 2+3=5", "100, 3+3=99")},
            logprob_ratios={"grpo-0-0": 1.0, "grpo-0-1": 1.0},
        )
        batch = policy.rollout(("math",), group_size=2)
        reward = NumericAnswerRewardAdapter({"math": "4"}).score(batch)

        self.assertEqual(reward.sample_rewards, {"grpo-0-0": 1.0, "grpo-0-1": 0.0})

    def test_run_grpo_step_returns_report_with_group_metrics(self):
        policy = MockGroupRolloutPolicy(
            completions={"math": ("4", "5")},
            logprob_ratios={"grpo-0-0": 1.1, "grpo-0-1": 0.9},
            kl_by_sample={"grpo-0-0": 0.1, "grpo-0-1": 0.2},
        )
        reward = ExactAnswerRewardAdapter({"math": "4"})

        result = run_grpo_step(("math",), policy, reward, GRPOConfig(group_size=2, kl_weight=0.1))

        self.assertEqual(result.batch.sample_ids, ("grpo-0-0", "grpo-0-1"))
        self.assertLess(result.objective.objective, 0.0)
        self.assertEqual(result.report.algorithm.name, "grpo")
        self.assertIn("group_size", result.report.metrics)


if __name__ == "__main__":
    unittest.main()
