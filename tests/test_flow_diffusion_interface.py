import unittest

from mindrl_repo.flow_diffusion_interface import (
    FlowDiffusionTrace,
    compare_flow_strategies,
    evaluate_flow_adapter,
    summarize_flow_trace,
)


class FlowDiffusionInterfaceTest(unittest.TestCase):
    def test_summarize_flow_trace_computes_barriers(self):
        summary = summarize_flow_trace(
            FlowDiffusionTrace(
                reward=1.0,
                surrogate_scores=(0.5, 1.0, 0.5),
                anchor_distances=(0.2, 0.4),
                smoothness_costs=(0.1, 0.3),
                collision_or_feasibility_failures=1,
            )
        )

        self.assertGreater(summary.surrogate_variance, 0.0)
        self.assertAlmostEqual(summary.drift, 0.3)
        self.assertAlmostEqual(summary.smoothness, 0.2)
        self.assertAlmostEqual(summary.failure_rate, 0.1)

    def test_high_drift_flow_uses_anchored_surrogate(self):
        result = evaluate_flow_adapter(
            FlowDiffusionTrace(
                reward=1.0,
                surrogate_scores=(0.2, 0.9, 0.4),
                anchor_distances=(0.6, 0.7),
                smoothness_costs=(0.2,),
            )
        )

        self.assertEqual(result.decision.adapter, "anchored_flow_surrogate")
        self.assertGreater(result.decision.anchor_strength, 1.0)
        self.assertLess(result.decision.branch_weight, 1.0)

    def test_compare_flow_strategies_returns_two_strategies(self):
        results = compare_flow_strategies(
            FlowDiffusionTrace(
                reward=1.0,
                surrogate_scores=(0.7, 0.8, 0.75),
                anchor_distances=(0.1, 0.2),
            )
        )

        self.assertEqual(set(results), {"score_routing", "barrier_gated"})


if __name__ == "__main__":
    unittest.main()
