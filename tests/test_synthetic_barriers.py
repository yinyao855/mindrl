import unittest

from mindrl.synthetic_barriers import (
    binary_pair_distribution,
    total_correlation,
    within_block_barrier,
)


class SyntheticBarriersTest(unittest.TestCase):
    def test_total_correlation_increases_with_pair_dependence(self):
        low = total_correlation(binary_pair_distribution(0.25))
        medium = total_correlation(binary_pair_distribution(0.40))
        high = total_correlation(binary_pair_distribution(0.48))

        self.assertLess(low, medium)
        self.assertLess(medium, high)

    def test_refining_blocks_reduces_within_block_barrier(self):
        joint = {
            (0, 0, 0): 0.46,
            (1, 1, 1): 0.46,
            (0, 0, 1): 0.08 / 6,
            (0, 1, 0): 0.08 / 6,
            (0, 1, 1): 0.08 / 6,
            (1, 0, 0): 0.08 / 6,
            (1, 0, 1): 0.08 / 6,
            (1, 1, 0): 0.08 / 6,
        }

        coarse = within_block_barrier(joint, blocks=[(0, 1, 2)])
        refined = within_block_barrier(joint, blocks=[(0,), (1, 2)])
        singleton = within_block_barrier(joint, blocks=[(0,), (1,), (2,)])

        self.assertGreater(coarse, refined)
        self.assertAlmostEqual(singleton, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
