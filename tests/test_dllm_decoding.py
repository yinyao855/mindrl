import unittest

from mindrl.dllm_decoding import (
    MockDLMDecoder,
    compare_fixed_and_adaptive,
    dllm_reproduction_command,
)


class DLMDecodingTest(unittest.TestCase):
    def test_adaptive_block_improves_high_uncertainty_mock_decoder(self):
        decoder = MockDLMDecoder(task="gsm8k", uncertainty=0.10)

        fixed, adaptive = compare_fixed_and_adaptive(
            decoder,
            "Question:",
            fixed_block_size=8,
            adaptive_b_min=1,
            adaptive_b_max=16,
            adaptive_alpha=0.4,
        )

        self.assertEqual(fixed.mode, "fixed")
        self.assertEqual(adaptive.mode, "adaptive")
        self.assertLess(adaptive.average_block_size, fixed.average_block_size)
        self.assertGreater(adaptive.score, fixed.score)

    def test_llada_command_template_mentions_model_and_task(self):
        command = dllm_reproduction_command(model="GSAI-ML/LLaDA-8B-Base", task="gsm8k")

        self.assertIn("accelerate launch eval_llada.py", command)
        self.assertIn("GSAI-ML/LLaDA-8B-Base", command)
        self.assertIn("--tasks gsm8k", command)


if __name__ == "__main__":
    unittest.main()
