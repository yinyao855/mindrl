import unittest

from mindrl_repo.agentic_barriers import AgentStep, summarize_trace_barriers


class AgenticBarriersTest(unittest.TestCase):
    def test_trace_summary_detects_invalid_tools_and_late_stop(self):
        trace = [
            AgentStep(kind="reasoning"),
            AgentStep(kind="tool_call", valid=True, latency=2.0, utility=0.6),
            AgentStep(kind="tool_call", valid=False, latency=4.0, utility=0.0),
            AgentStep(kind="delegate", utility=0.1),
            AgentStep(kind="delegate", utility=0.0),
            AgentStep(kind="stop", stop_should_have_happened_at=3),
        ]

        summary = summarize_trace_barriers(trace)

        self.assertEqual(summary.tool_calls, 2)
        self.assertEqual(summary.invalid_tool_calls, 1)
        self.assertAlmostEqual(summary.mean_tool_latency, 3.0)
        self.assertEqual(summary.redundant_delegations, 1)
        self.assertEqual(summary.late_stop_steps, 2)


if __name__ == "__main__":
    unittest.main()
