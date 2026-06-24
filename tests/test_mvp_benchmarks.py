import json
import tempfile
import unittest
from pathlib import Path

from mindrl.mvp_benchmarks import (
    build_ar_grpo_smoke_report,
    build_ar_lora_plan_report,
    build_ar_opd_smoke_report,
    build_controller_ablation_report,
    build_diffusion_ddpo_smoke_report,
    write_reports,
)


class MVPBenchmarksTest(unittest.TestCase):
    def test_ar_grpo_smoke_report_contains_expected_metrics(self):
        report = build_ar_grpo_smoke_report()
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "grpo")
        self.assertEqual(record["algorithm"]["branch"], "ar")
        self.assertIn("reward_mean", record["metrics"])
        self.assertIn("objective", record["metrics"])

    def test_diffusion_ddpo_smoke_report_contains_expected_metrics(self):
        report = build_diffusion_ddpo_smoke_report()
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "ddpo")
        self.assertEqual(record["algorithm"]["branch"], "diffusion")
        self.assertIn("branch_weight", record["metrics"])
        self.assertIn("image_grid_manifest", record["artifacts"])

    def test_controller_ablation_report_contains_strategy_metrics(self):
        report = build_controller_ablation_report()
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "controller_ablation")
        self.assertIn("barrier_gated_risk_cost", record["metrics"])
        self.assertIn("best_strategy", record["artifacts"])

    def test_ar_opd_smoke_report_contains_clipping_metrics(self):
        report = build_ar_opd_smoke_report()
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "opd")
        self.assertEqual(record["algorithm"]["branch"], "ar")
        self.assertIn("clipped_tokens", record["metrics"])
        self.assertIn("mean_teacher_entropy", record["metrics"])

    def test_ar_lora_plan_report_contains_resource_estimate(self):
        report = build_ar_lora_plan_report("qwen-7b")
        record = report.to_json_record()

        self.assertEqual(record["algorithm"]["name"], "ar_lora")
        self.assertIn("estimated_vram_gb", record["metrics"])
        self.assertEqual(record["artifacts"]["model"], "Qwen/Qwen2.5-7B")

    def test_write_reports_creates_jsonl_and_markdown(self):
        reports = (
            build_ar_grpo_smoke_report(),
            build_ar_opd_smoke_report(),
            build_ar_lora_plan_report("qwen-7b"),
            build_diffusion_ddpo_smoke_report(),
            build_controller_ablation_report(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_reports(Path(tmpdir), reports)

            jsonl_lines = paths["jsonl"].read_text(encoding="utf-8").strip().splitlines()
            markdown = paths["markdown"].read_text(encoding="utf-8")

        self.assertEqual(len(jsonl_lines), 5)
        self.assertEqual(json.loads(jsonl_lines[0])["algorithm"]["name"], "grpo")
        self.assertIn("ar-grpo-smoke", markdown)
        self.assertIn("ar-opd-smoke", markdown)
        self.assertIn("ar-lora-plan-qwen-7b", markdown)
        self.assertIn("diffusion-ddpo-smoke", markdown)
        self.assertIn("controller-ablation-smoke", markdown)


if __name__ == "__main__":
    unittest.main()
