import unittest

from mindrl_repo.interface_controller import (
    BarrierProfile,
    MindRLController,
    PolicySpec,
)


class InterfaceControllerTest(unittest.TestCase):
    def test_ar_branch_keeps_exact_ratio(self):
        decision = MindRLController().decide(
            PolicySpec(name="reasoning", modality="text", structure="ar", score_availability="exact"),
            BarrierProfile(),
        )

        self.assertEqual(decision.adapter, "exact_ratio")
        self.assertEqual(decision.branch_weight, 1.0)

    def test_high_nctc_parallel_discrete_branch_shrinks_block(self):
        decision = MindRLController().decide(
            PolicySpec(
                name="masked_tokens",
                modality="text",
                structure="parallel_discrete",
                score_availability="marginal",
                default_granularity=16,
            ),
            BarrierProfile(nctc=0.45),
        )

        self.assertEqual(decision.adapter, "dependence_aware_block")
        self.assertEqual(decision.granularity, 4)
        self.assertGreater(decision.structure_cost_weight, 0.0)

    def test_high_flow_drift_strengthens_anchor_and_reduces_weight(self):
        decision = MindRLController().decide(
            PolicySpec(
                name="action_chunk",
                modality="continuous_action",
                structure="flow",
                score_availability="surrogate",
            ),
            BarrierProfile(surrogate_variance=0.30, drift=0.55, smoothness=0.20),
        )

        self.assertEqual(decision.adapter, "anchored_flow_surrogate")
        self.assertGreater(decision.anchor_strength, 1.0)
        self.assertLess(decision.branch_weight, 1.0)
        self.assertEqual(decision.clip_range, 0.1)


if __name__ == "__main__":
    unittest.main()
