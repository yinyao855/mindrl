import math
import unittest

from mindrl_repo.discrete_interface import (
    adaptive_block_size,
    estimate_nctc_from_logprobs,
    sample_distance_controlled_block,
)


class DiscreteInterfaceTest(unittest.TestCase):
    def test_distance_controlled_sampler_respects_gap_and_seed(self):
        block = sample_distance_controlled_block(
            eligible_positions=list(range(10)),
            block_size=4,
            gap=3,
            seed=0,
        )

        self.assertEqual(block, [7, 1, 4])
        self.assertTrue(
            all(abs(i - j) >= 3 for idx, i in enumerate(block) for j in block[idx + 1 :])
        )

    def test_nctc_estimator_matches_pair_normalized_log_gap(self):
        joint_order_logprobs = [
            [-0.2, -0.3, -0.4],
            [-0.25, -0.35, -0.45],
        ]
        marginal_logprobs = [-0.5, -0.7, -0.9]

        nctc = estimate_nctc_from_logprobs(joint_order_logprobs, marginal_logprobs)

        # mean joint logprob = -0.975, marginal sum = -2.1, D = 1.125.
        # Pair normalization for B=3 divides by C(3, 2)=3.
        self.assertTrue(math.isclose(nctc, 0.375, rel_tol=1e-9))

    def test_adaptive_block_size_shrinks_as_uncertainty_rises(self):
        low_uncertainty = adaptive_block_size(
            [0.05, 0.10, 0.08], alpha=1.0, b_min=1, b_max=16
        )
        high_uncertainty = adaptive_block_size(
            [0.8, 0.9, 1.0], alpha=1.0, b_min=1, b_max=16
        )

        self.assertEqual(low_uncertainty, 13)
        self.assertEqual(high_uncertainty, 1)
        self.assertGreater(low_uncertainty, high_uncertainty)


if __name__ == "__main__":
    unittest.main()
