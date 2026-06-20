import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TaskNCTCCLITest(unittest.TestCase):
    def test_mock_cli_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/run_task_nctc_proxy.py",
                    "--sample",
                    "preset",
                    "--max-examples",
                    "3",
                    "--device",
                    "cpu",
                    "--mock-scorer",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertIn("dependency_group_summary", result.stdout)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["task"], "gsm8k_smoke")
        self.assertIn("pair_normalized_gap", records[0])

    def test_block_level_mock_cli_writes_nctc_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "block_records.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/run_task_nctc_proxy.py",
                    "--sample",
                    "preset",
                    "--max-examples",
                    "2",
                    "--device",
                    "cpu",
                    "--mock-scorer",
                    "--block-size",
                    "2",
                    "--block-stride",
                    "2",
                    "--order-count",
                    "2",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            records = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertIn("mean_pair_normalized_nctc", result.stdout)
        self.assertGreaterEqual(len(records), 2)
        self.assertIn("pair_normalized_nctc", records[0])
        self.assertEqual(records[0]["order_count"], 2)


if __name__ == "__main__":
    unittest.main()
