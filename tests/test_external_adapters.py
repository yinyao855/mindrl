import unittest

from mindrl.ar_trainer import qwen_lora_preset
from mindrl.external_adapters import (
    export_openrlhf_args,
    export_trl_grpo_config,
    export_verl_opd_config,
)
from mindrl.grpo import GRPOConfig
from mindrl.opd import OPDConfig


class ExternalAdaptersTest(unittest.TestCase):
    def test_export_trl_grpo_config_maps_group_and_lora_plan(self):
        config = export_trl_grpo_config(
            qwen_lora_preset("qwen-7b"),
            GRPOConfig(group_size=8, kl_weight=0.02),
        )

        self.assertEqual(config["model_name_or_path"], "Qwen/Qwen2.5-7B")
        self.assertEqual(config["num_generations"], 8)
        self.assertEqual(config["beta"], 0.02)
        self.assertEqual(config["lora_r"], 16)
        self.assertTrue(config["load_in_4bit"])

    def test_export_verl_opd_config_maps_teacher_student_controls(self):
        config = export_verl_opd_config(
            qwen_lora_preset("qwen-1.5b"),
            OPDConfig(per_token_clip=0.25),
            teacher_model="Qwen/Qwen2.5-7B",
        )

        self.assertEqual(config["algorithm"], "opd")
        self.assertEqual(config["student_model"], "Qwen/Qwen2.5-1.5B")
        self.assertEqual(config["teacher_model"], "Qwen/Qwen2.5-7B")
        self.assertAlmostEqual(config["distillation"]["per_token_clip"], 0.25)

    def test_export_openrlhf_args_returns_command_arguments(self):
        args = export_openrlhf_args(
            qwen_lora_preset("qwen-7b"),
            GRPOConfig(group_size=4, kl_weight=0.01),
        )

        self.assertIn("--pretrain Qwen/Qwen2.5-7B", args)
        self.assertIn("--algo.advantage.estimator group_norm", args)
        self.assertIn("--lora_rank 16", args)


if __name__ == "__main__":
    unittest.main()
