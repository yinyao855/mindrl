import unittest

from mindrl.diffusion_adapter import (
    MockDiffusionPipeline,
    build_image_grid_manifest,
    collect_diffusion_rollouts,
)


class DiffusionAdapterTest(unittest.TestCase):
    def test_mock_pipeline_generates_images_and_trajectory_stats(self):
        pipeline = MockDiffusionPipeline(
            images={"red cat": "cat-image"},
            captions={"red cat": "a red cat"},
            step_logprobs={"red cat": (-0.1, -0.2)},
            anchor_distances={"red cat": 0.3},
        )

        result = pipeline.generate("red cat", sample_id="img-0")

        self.assertEqual(result.sample_id, "img-0")
        self.assertEqual(result.image, "cat-image")
        self.assertEqual(result.trajectory.step_logprobs, (-0.1, -0.2))
        self.assertEqual(result.trajectory.image_caption, "a red cat")

    def test_collect_diffusion_rollouts_returns_trajectories_and_images(self):
        pipeline = MockDiffusionPipeline(
            images={"red cat": "cat-image", "blue dog": "dog-image"},
            captions={"red cat": "a red cat", "blue dog": "a blue dog"},
            step_logprobs={"red cat": (-0.1,), "blue dog": (-0.2,)},
            anchor_distances={"red cat": 0.1, "blue dog": 0.2},
        )

        batch = collect_diffusion_rollouts(("red cat", "blue dog"), pipeline)

        self.assertEqual(tuple(t.sample_id for t in batch.trajectories), ("diff-0", "diff-1"))
        self.assertEqual(batch.images["diff-0"], "cat-image")
        self.assertEqual(batch.captions["diff-1"], "a blue dog")

    def test_build_image_grid_manifest_lists_samples_for_report(self):
        pipeline = MockDiffusionPipeline(
            images={"red cat": "cat-image"},
            captions={"red cat": "a red cat"},
            step_logprobs={"red cat": (-0.1,)},
            anchor_distances={"red cat": 0.1},
        )
        batch = collect_diffusion_rollouts(("red cat",), pipeline)

        manifest = build_image_grid_manifest(batch)

        self.assertIn("diff-0", manifest)
        self.assertIn("red cat", manifest)
        self.assertIn("a red cat", manifest)


if __name__ == "__main__":
    unittest.main()
