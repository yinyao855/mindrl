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

    def test_harder_math_smoke_examples_include_multi_step_arithmetic(self):
        import mindrl.smoke_prompts as smoke_prompts

        examples = smoke_prompts.harder_math_smoke_examples()

        self.assertGreaterEqual(len(examples), 4)
        self.assertIn("12 * 7", examples[0].prompt)
        self.assertEqual(examples[0].target, "84")

    def test_math_smoke_prompts_and_answers_can_select_harder_set(self):
        import mindrl.smoke_prompts as smoke_prompts

        prompts, answers = smoke_prompts.math_smoke_prompts_and_answers("harder")

        self.assertGreaterEqual(len(prompts), 4)
        self.assertEqual(answers[prompts[0]], "84")


if __name__ == "__main__":
    unittest.main()
