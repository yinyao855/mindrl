import unittest

from mindrl.diffusion_training import (
    DiffusionTrajectory,
    clip_alignment_reward,
    compressibility_reward,
    compute_ddpo_objective,
    summarize_diffusion_run,
)


class DiffusionTrainingTest(unittest.TestCase):
    def test_compressibility_reward_prefers_short_serialized_images(self):
        rewards = compressibility_reward(
            {
                "img-small": "aaa",
                "img-large": "aaaaaaaaaa",
            }
        )

        self.assertGreater(rewards.sample_rewards["img-small"], rewards.sample_rewards["img-large"])

    def test_clip_alignment_reward_scores_prompt_token_overlap(self):
        rewards = clip_alignment_reward(
            prompts={"s1": "red cat", "s2": "blue dog"},
            image_captions={"s1": "a red cat sitting", "s2": "a green bird"},
        )

        self.assertAlmostEqual(rewards.sample_rewards["s1"], 1.0)
        self.assertAlmostEqual(rewards.sample_rewards["s2"], 0.0)

    def test_ddpo_objective_uses_reward_advantage_and_denoising_logprob(self):
        trajectories = (
            DiffusionTrajectory("s1", "cat", step_logprobs=(-0.1, -0.2), anchor_distance=0.1),
            DiffusionTrajectory("s2", "dog", step_logprobs=(-0.4, -0.3), anchor_distance=0.3),
        )
        rewards = compressibility_reward({"s1": "aa", "s2": "aaaaaaaa"})

        objective = compute_ddpo_objective(trajectories, rewards, kl_anchor_weight=0.1)

        self.assertGreater(objective.sample_weights["s1"], 0.0)
        self.assertLess(objective.sample_weights["s2"], 0.0)
        self.assertIn("anchor_penalty", objective.diagnostics)

    def test_summarize_diffusion_run_emits_controller_decision(self):
        trajectories = (
            DiffusionTrajectory("s1", "cat", step_logprobs=(-0.1, -0.2), anchor_distance=0.5),
            DiffusionTrajectory("s2", "cat", step_logprobs=(-0.3, -0.4), anchor_distance=0.6),
        )
        rewards = compressibility_reward({"s1": "aa", "s2": "aaaa"})

        report = summarize_diffusion_run("ddpo-smoke", trajectories, rewards)
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "ddpo")
        self.assertIn("reward_mean", record["metrics"])
        self.assertIn("adapter", record["artifacts"])


if __name__ == "__main__":
    unittest.main()
