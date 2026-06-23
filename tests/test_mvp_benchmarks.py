import json
import tempfile
import unittest
from pathlib import Path

from mindrl.mvp_benchmarks import (
    build_ar_grpo_smoke_report,
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

    def test_write_reports_creates_jsonl_and_markdown(self):
        reports = (build_ar_grpo_smoke_report(), build_diffusion_ddpo_smoke_report())
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_reports(Path(tmpdir), reports)

            jsonl_lines = paths["jsonl"].read_text(encoding="utf-8").strip().splitlines()
            markdown = paths["markdown"].read_text(encoding="utf-8")

        self.assertEqual(len(jsonl_lines), 2)
        self.assertEqual(json.loads(jsonl_lines[0])["algorithm"]["name"], "grpo")
        self.assertIn("ar-grpo-smoke", markdown)
        self.assertIn("diffusion-ddpo-smoke", markdown)


if __name__ == "__main__":
    unittest.main()
