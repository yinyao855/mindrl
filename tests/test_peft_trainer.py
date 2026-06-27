import unittest

import torch

import mindrl.peft_trainer as peft_trainer
from mindrl.peft_trainer import (
    SFTExample,
    build_sft_text,
    summarize_update,
)


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

    def test_clipped_opd_loss_tensor_limits_large_teacher_gap(self):
        if not hasattr(peft_trainer, "clipped_opd_loss_tensor"):
            self.fail("clipped_opd_loss_tensor is not implemented")
        loss = peft_trainer.clipped_opd_loss_tensor(
            student_logprobs=torch.tensor([-1.3, -0.4, -0.3]),
            teacher_logprobs=torch.tensor([-0.1, -0.2, -0.2]),
            per_token_clip=0.25,
        )

        self.assertAlmostEqual(float(loss.detach()), 0.18333333730697632)

    def test_summarize_opd_update_reports_clipping_and_rewards(self):
        if not hasattr(peft_trainer, "summarize_opd_update"):
            self.fail("summarize_opd_update is not implemented")
        report = peft_trainer.summarize_opd_update(
            run_name="opd-update",
            model_name="tiny",
            before_loss=0.3,
            after_loss=0.2,
            raw_objective=0.5,
            clipped_objective=0.2,
            clipped_token_ratio=0.25,
            mean_teacher_entropy=1.5,
            before_reward=0.0,
            after_reward=0.5,
            trainable_parameters=64,
        )
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "peft_opd")
        self.assertAlmostEqual(record["metrics"]["loss_delta"], -0.1)
        self.assertAlmostEqual(record["metrics"]["clipped_token_ratio"], 0.25)
        self.assertAlmostEqual(record["metrics"]["reward_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
