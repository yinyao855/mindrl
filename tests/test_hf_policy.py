import unittest

from mindrl.hf_policy import (
    GenerationRecord,
    format_privileged_context,
    records_to_rollout_batch,
)


class HFPolicyHelpersTest(unittest.TestCase):
    def test_records_to_rollout_batch_preserves_prompt_group_metadata(self):
        records = (
            GenerationRecord(
                sample_id="hf-0-0",
                prompt="2+2=",
                response="4",
                token_logprobs=(-0.1,),
            ),
            GenerationRecord(
                sample_id="hf-0-1",
                prompt="2+2=",
                response="5",
                token_logprobs=(-0.2,),
            ),
        )

        batch = records_to_rollout_batch(records)

        self.assertEqual(batch.sample_ids, ("hf-0-0", "hf-0-1"))
        self.assertEqual(batch.samples[0].metadata["prompt_id"], "2+2=")
        self.assertEqual(batch.samples[1].metadata["group_index"], 1)

    def test_format_privileged_context_includes_answer_without_response(self):
        context = format_privileged_context("2+2=", "4")

        self.assertIn("Correct answer: 4", context)
        self.assertIn("Problem: 2+2=", context)
        self.assertTrue(context.endswith("Student response:"))

    def test_group_policy_interface_allows_single_sample_rollout(self):
        class MinimalPolicy:
            generate_records = lambda self, prompts, group_size=1: (
                GenerationRecord("hf-0-0", prompts[0], "4", (-0.1,)),
            )

            from mindrl.hf_policy import HFCausalLMGroupPolicy

            rollout = HFCausalLMGroupPolicy.rollout

        batch = MinimalPolicy().rollout(("2+2=",))

        self.assertEqual(batch.sample_ids, ("hf-0-0",))

    def test_group_policy_score_returns_token_logprobs(self):
        class MinimalPolicy:
            def generate_records(self, prompts, group_size=1):
                self._records = (GenerationRecord("hf-0-0", prompts[0], "4", (-0.1,)),)
                return self._records

            from mindrl.hf_policy import HFCausalLMGroupPolicy

            rollout = HFCausalLMGroupPolicy.rollout
            score = HFCausalLMGroupPolicy.score

        policy = MinimalPolicy()
        batch = policy.rollout(("2+2=",))

        self.assertEqual(policy.score(batch), {"hf-0-0": (-0.1,)})


if __name__ == "__main__":
    unittest.main()
