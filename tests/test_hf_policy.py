import unittest

from mindrl.hf_policy import (
    GenerationRecord,
    format_privileged_context,
    records_to_rollout_batch,
)
import mindrl.hf_policy as hf_policy


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
        object.__setattr__(records[0], "reference_token_logprobs", (-0.2,))
        object.__setattr__(records[1], "reference_token_logprobs", (-0.1,))

        batch = records_to_rollout_batch(records)

        self.assertEqual(batch.sample_ids, ("hf-0-0", "hf-0-1"))
        self.assertEqual(batch.samples[0].metadata["prompt_id"], "2+2=")
        self.assertEqual(batch.samples[1].metadata["group_index"], 1)
        self.assertEqual(batch.samples[0].metadata.get("reference_token_count"), 1)

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

    def test_group_policy_ratio_and_kl_are_record_backed(self):
        class MinimalPolicy:
            def generate_records(self, prompts, group_size=1):
                self._records = (
                    GenerationRecord("hf-0-0", prompts[0], "4", (-0.1,)),
                    GenerationRecord("hf-0-1", prompts[0], "5", (-0.4,)),
                )
                object.__setattr__(self._records[0], "reference_token_logprobs", (-0.3,))
                object.__setattr__(self._records[1], "reference_token_logprobs", (-0.2,))
                return self._records

            from mindrl.hf_policy import HFCausalLMGroupPolicy

            rollout = HFCausalLMGroupPolicy.rollout
            logprob_ratios = HFCausalLMGroupPolicy.logprob_ratios
            kl = HFCausalLMGroupPolicy.kl

        policy = MinimalPolicy()
        batch = policy.rollout(("2+2=",), group_size=2)

        self.assertGreater(policy.logprob_ratios(batch)["hf-0-0"], 1.0)
        self.assertLess(policy.logprob_ratios(batch)["hf-0-1"], 1.0)
        self.assertAlmostEqual(policy.kl(batch)["hf-0-0"], 0.2)

    def test_resolve_torch_dtype_supports_large_model_loading_modes(self):
        import torch

        if not hasattr(hf_policy, "resolve_torch_dtype"):
            self.fail("resolve_torch_dtype is not implemented")
        self.assertIs(hf_policy.resolve_torch_dtype("fp16", torch.device("cuda")), torch.float16)
        self.assertIs(hf_policy.resolve_torch_dtype("bf16", torch.device("cuda")), torch.bfloat16)
        self.assertIs(hf_policy.resolve_torch_dtype("fp32", torch.device("cpu")), torch.float32)
        self.assertIsNone(hf_policy.resolve_torch_dtype("auto", torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
