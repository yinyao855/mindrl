import unittest

from mindrl.peft_trainer import SFTExample, build_sft_text, summarize_update


class PeftTrainerHelpersTest(unittest.TestCase):
    def test_build_sft_text_appends_eos(self):
        text = build_sft_text(SFTExample(prompt="2+2=", target="4"), eos_token="<eos>")

        self.assertEqual(text, "2+2=4<eos>")

    def test_summarize_update_reports_before_after_metrics(self):
        report = summarize_update(
            run_name="tiny-update",
            model_name="tiny",
            before_loss=2.0,
            after_loss=1.0,
            before_reward=0.0,
            after_reward=0.5,
            trainable_parameters=128,
        )
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "peft_sft")
        self.assertAlmostEqual(record["metrics"]["loss_delta"], -1.0)
        self.assertAlmostEqual(record["metrics"]["reward_delta"], 0.5)
        self.assertEqual(record["metrics"]["trainable_parameters"], 128.0)


if __name__ == "__main__":
    unittest.main()
