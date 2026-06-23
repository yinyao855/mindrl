import unittest

from mindrl.ar_proxy_nctc import (
    bootstrap_mean_ci,
    build_block_nctc_record,
    build_proxy_record,
    default_block_starts,
    pair_normalized_dependency_gap,
    summarize_proxy_records,
)


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

    def test_build_proxy_record_stores_logprob_sums_and_gap(self):
        record = build_proxy_record(
            task="gsm8k_smoke",
            dependency_group="high",
            joint_chain_logprobs=[-0.2, -0.3],
            prompt_only_marginal_logprobs=[-1.0, -1.2],
        )

        self.assertEqual(record.task, "gsm8k_smoke")
        self.assertEqual(record.dependency_group, "high")
        self.assertEqual(record.token_count, 2)
        self.assertAlmostEqual(record.joint_logprob, -0.5)
        self.assertAlmostEqual(record.marginal_logprob, -2.2)
        self.assertAlmostEqual(record.pair_normalized_gap, 1.7)

    def test_summarize_proxy_records_groups_by_dependency_group(self):
        records = [
            build_proxy_record("a", "high", [-0.5, -0.5], [-1.0, -1.0]),
            build_proxy_record("b", "high", [-0.8, -0.8], [-1.0, -1.0]),
            build_proxy_record("c", "low", [-0.9, -0.9], [-1.0, -1.0]),
        ]

        summaries = summarize_proxy_records(records, group_by="dependency_group")

        self.assertEqual([summary.group for summary in summaries], ["high", "low"])
        self.assertEqual([summary.count for summary in summaries], [2, 1])
        self.assertAlmostEqual(summaries[0].mean_pair_normalized_gap, 0.7)
        self.assertAlmostEqual(summaries[1].mean_pair_normalized_gap, 0.2)

    def test_build_block_nctc_record_averages_multiple_orders(self):
        record = build_block_nctc_record(
            task="gsm8k",
            dependency_group="high",
            block_start=4,
            joint_order_logprobs=[[-0.4, -0.5, -0.6], [-0.6, -0.5, -0.4]],
            marginal_logprobs=[-1.0, -1.1, -1.2],
        )

        self.assertEqual(record.block_start, 4)
        self.assertEqual(record.token_count, 3)
        self.assertEqual(record.order_count, 2)
        self.assertAlmostEqual(record.mean_joint_logprob, -1.5)
        self.assertAlmostEqual(record.marginal_logprob, -3.3)
        self.assertAlmostEqual(record.pair_normalized_nctc, 1.8 / 3)

    def test_default_block_starts_uses_stride(self):
        self.assertEqual(default_block_starts(10, block_size=4, stride=3), [0, 3, 6])
        self.assertEqual(default_block_starts(3, block_size=4), [])

    def test_bootstrap_mean_ci_is_deterministic(self):
        ci = bootstrap_mean_ci([1.0, 2.0, 3.0], samples=20, seed=7)

        self.assertAlmostEqual(ci.mean, 2.0)
        self.assertEqual(ci.samples, 20)
        self.assertLessEqual(ci.low, ci.mean)
        self.assertGreaterEqual(ci.high, ci.mean)


if __name__ == "__main__":
    unittest.main()
