import unittest

from mindrl.ar_trainer import (
    ARModelConfig,
    LoRAConfig,
    TrainerConfig,
    build_ar_trainer_plan,
    qwen_lora_preset,
    summarize_ar_trainer_plan,
)


class ARTrainerConfigTest(unittest.TestCase):
    def test_qwen_lora_preset_sets_model_scale_defaults(self):
        small = qwen_lora_preset("qwen-0.5b")
        large = qwen_lora_preset("qwen-13b")

        self.assertEqual(small.model.parameter_count_b, 0.5)
        self.assertEqual(large.model.parameter_count_b, 13.0)
        self.assertTrue(large.trainer.gradient_checkpointing)
        self.assertEqual(large.lora.quantization, "4bit")

    def test_build_ar_trainer_plan_estimates_memory_and_steps(self):
        plan = build_ar_trainer_plan(
            ARModelConfig(name="Qwen/Qwen2.5-7B", parameter_count_b=7.0),
            LoRAConfig(rank=16, alpha=32, quantization="4bit"),
            TrainerConfig(
                per_device_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
            ),
        )

        self.assertEqual(plan.effective_batch_size, 8)
        self.assertGreater(plan.estimated_vram_gb, 0.0)
        self.assertIn("lora", plan.tags)
        self.assertIn("qlora", plan.tags)

    def test_summarize_ar_trainer_plan_returns_report(self):
        preset = qwen_lora_preset("qwen-7b")

        report = summarize_ar_trainer_plan("ar-lora-smoke", preset)
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "ar_lora")
        self.assertEqual(record["algorithm"]["branch"], "ar")
        self.assertIn("estimated_vram_gb", record["metrics"])
        self.assertEqual(record["artifacts"]["model"], "Qwen/Qwen2.5-7B")


if __name__ == "__main__":
    unittest.main()
