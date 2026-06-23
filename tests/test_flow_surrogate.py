import unittest

from mindrl.flow_surrogate import (
    FlowSurrogateSetting,
    evaluate_flow_surrogate,
    sweep_anchor_strengths,
)


class FlowSurrogateTest(unittest.TestCase):
    def test_anchor_reduces_residual_drift(self):
        loose = evaluate_flow_surrogate(
            FlowSurrogateSetting(1.0, 0.3, 0.6, anchor_strength=0.0, clip_range=0.2)
        )
        anchored = evaluate_flow_surrogate(
            FlowSurrogateSetting(1.0, 0.3, 0.6, anchor_strength=2.0, clip_range=0.2)
        )

        self.assertLess(anchored.residual_drift, loose.residual_drift)
        self.assertGreater(anchored.risk_adjusted_objective, loose.risk_adjusted_objective)

    def test_smaller_clip_reduces_update_magnitude(self):
        clipped = evaluate_flow_surrogate(
            FlowSurrogateSetting(1.0, 0.3, 0.6, anchor_strength=2.0, clip_range=0.1)
        )

        self.assertAlmostEqual(clipped.effective_update, 0.5)

    def test_sweep_anchor_strengths_preserves_order(self):
        results = sweep_anchor_strengths(1.0, 0.3, 0.6, [0.0, 1.0, 2.0], 0.2)

        self.assertEqual(len(results), 3)
        self.assertGreater(results[0].residual_drift, results[-1].residual_drift)


if __name__ == "__main__":
    unittest.main()
