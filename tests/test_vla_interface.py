import unittest

from mindrl_repo.flow_diffusion_interface import FlowDiffusionTrace
from mindrl_repo.vla_interface import (
    ARBranchTrace,
    VLABranchTrace,
    compare_vla_strategies,
    evaluate_vla_controller,
)


class VLAInterfaceTest(unittest.TestCase):
    def test_vla_controller_allocates_semantic_refresh_budget(self):
        result = evaluate_vla_controller(
            VLABranchTrace(
                task_success_proxy=0.6,
                ar=ARBranchTrace(token_reward=0.2),
                flow=FlowDiffusionTrace(
                    reward=0.8,
                    surrogate_scores=(0.4, 0.9, 0.6),
                    anchor_distances=(0.5, 0.6),
                    smoothness_costs=(0.2,),
                ),
                semantic_staleness=0.25,
                world_uncertainty=0.35,
            )
        )

        self.assertEqual(result.ar_decision.adapter, "exact_ratio")
        self.assertEqual(result.flow_decision.adapter, "anchored_flow_surrogate")
        self.assertAlmostEqual(result.semantic_refresh_budget, 0.60)

    def test_compare_vla_strategies_returns_baseline_and_gated(self):
        trace = VLABranchTrace(
            task_success_proxy=0.6,
            ar=ARBranchTrace(token_reward=0.2),
            flow=FlowDiffusionTrace(
                reward=0.8,
                surrogate_scores=(0.4, 0.9, 0.6),
                anchor_distances=(0.5, 0.6),
            ),
            semantic_staleness=0.1,
            world_uncertainty=0.1,
        )

        results = compare_vla_strategies(trace)

        self.assertEqual(set(results), {"score_routing_only", "barrier_gated"})


if __name__ == "__main__":
    unittest.main()
