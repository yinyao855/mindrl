import unittest

from mindrl_repo.ar_proxy_nctc import pair_normalized_dependency_gap


class ARProxyNCTCTest(unittest.TestCase):
    def test_pair_normalized_dependency_gap_uses_joint_minus_prompt_marginals(self):
        joint_chain_logprobs = [-0.2, -0.3, -0.4, -0.5]
        prompt_only_marginal_logprobs = [-1.0, -1.1, -1.2, -1.3]

        score = pair_normalized_dependency_gap(
            joint_chain_logprobs,
            prompt_only_marginal_logprobs,
        )

        self.assertAlmostEqual(score, 3.2 / 6)

    def test_pair_normalized_dependency_gap_is_zero_for_singleton(self):
        score = pair_normalized_dependency_gap([-0.2], [-1.0])

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
