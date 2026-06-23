import unittest

from mindrl.core import (
    AlgorithmConfig,
    RewardOutput,
    RolloutBatch,
    RolloutSample,
    TeacherSignal,
    TrainReport,
)


class CoreAbstractionsTest(unittest.TestCase):
    def test_rollout_batch_groups_samples_by_branch(self):
        batch = RolloutBatch(
            samples=(
                RolloutSample(
                    sample_id="s1",
                    prompt="What is 2+2?",
                    response="4",
                    branch="ar",
                    metadata={"task": "math"},
                ),
                RolloutSample(
                    sample_id="s2",
                    prompt="draw a cat",
                    response="image://cat",
                    branch="diffusion",
                    metadata={"task": "image"},
                ),
            )
        )

        self.assertEqual(batch.sample_ids, ("s1", "s2"))
        self.assertEqual(batch.by_branch("ar")[0].response, "4")
        self.assertEqual(batch.by_branch("diffusion")[0].metadata["task"], "image")

    def test_rollout_batch_rejects_duplicate_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            RolloutBatch(
                samples=(
                    RolloutSample("s1", "p1", "r1", "ar"),
                    RolloutSample("s1", "p2", "r2", "ar"),
                )
            )

    def test_reward_output_aligns_with_rollout_batch(self):
        batch = RolloutBatch(
            samples=(
                RolloutSample("s1", "p1", "r1", "ar"),
                RolloutSample("s2", "p2", "r2", "ar"),
            )
        )
        rewards = RewardOutput(sample_rewards={"s1": 1.0, "s2": 0.0})

        rewards.validate_for(batch)

        self.assertAlmostEqual(rewards.mean_reward, 0.5)

    def test_teacher_signal_reports_token_statistics(self):
        signal = TeacherSignal(
            sample_id="s1",
            token_logprobs=(-0.1, -0.3, -0.2),
            topk_tokens=(("A", "B"), ("C",), ("D", "E")),
        )

        self.assertEqual(signal.token_count, 3)
        self.assertAlmostEqual(signal.mean_logprob, -0.2)

    def test_train_report_serializes_metrics_and_markdown(self):
        report = TrainReport(
            run_name="ar-grpo-smoke",
            algorithm=AlgorithmConfig(name="grpo", branch="ar", hyperparameters={"group_size": 4}),
            metrics={"reward_mean": 0.75, "kl": 0.02},
            artifacts={"jsonl": "outputs/run.jsonl"},
        )

        record = report.to_json_record()
        markdown = report.to_markdown()

        self.assertEqual(record["run_name"], "ar-grpo-smoke")
        self.assertEqual(record["algorithm"]["name"], "grpo")
        self.assertIn("reward_mean", markdown)
        self.assertIn("outputs/run.jsonl", markdown)


if __name__ == "__main__":
    unittest.main()
