import unittest


class SmokePromptsTest(unittest.TestCase):
    def test_math_smoke_examples_use_strict_numeric_answer_format(self):
        try:
            import mindrl.smoke_prompts as smoke_prompts
        except ModuleNotFoundError:
            self.fail("mindrl.smoke_prompts is not implemented")
        if not hasattr(smoke_prompts, "math_smoke_examples"):
            self.fail("math_smoke_examples is not implemented")

        examples = smoke_prompts.math_smoke_examples()

        self.assertEqual(examples[0].target, "4")
        self.assertIn("Question: What is 2 + 2?", examples[0].prompt)
        self.assertTrue(examples[0].prompt.endswith("Answer with only one number:\n"))


if __name__ == "__main__":
    unittest.main()
