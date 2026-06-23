import unittest

from mindrl.controller_ablation import BranchScenario, run_controller_ablation
from mindrl.interface_controller import BarrierProfile, PolicySpec


class ControllerAblationTest(unittest.TestCase):
    def test_barrier_gated_reduces_high_risk_update_budget(self):
        scenarios = [
            BranchScenario(
                spec=PolicySpec(
                    name="masked",
                    modality="text",
                    structure="parallel_discrete",
                    score_availability="marginal",
                    default_granularity=16,
                ),
                profile=BarrierProfile(nctc=0.50),
                reward=1.0,
            ),
            BranchScenario(
                spec=PolicySpec(
                    name="flow",
                    modality="action",
                    structure="flow",
                    score_availability="surrogate",
                    default_granularity=8,
                ),
                profile=BarrierProfile(surrogate_variance=0.40, drift=0.60),
                reward=1.0,
            ),
        ]

        uniform = run_controller_ablation(scenarios, "uniform")
        gated = run_controller_ablation(scenarios, "barrier_gated")

        self.assertLess(gated.update_budget, uniform.update_budget)
        self.assertLess(gated.risk_cost, uniform.risk_cost)
        self.assertIn("dependence_aware_block", [d.adapter for d in gated.decisions])


if __name__ == "__main__":
    unittest.main()
