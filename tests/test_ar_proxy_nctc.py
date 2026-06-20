import unittest

from mindrl_repo.ar_proxy_nctc import (
    build_proxy_record,
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


if __name__ == "__main__":
    unittest.main()
