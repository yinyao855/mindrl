import json
import tempfile
import unittest
from pathlib import Path

from mindrl.benchmark_suites import (
    load_curated_benchmark_samples,
    write_samples_jsonl,
)


class BenchmarkSuitesTest(unittest.TestCase):
    def test_curated_suite_covers_required_task_families(self):
        samples = load_curated_benchmark_samples()

        tasks = {sample.task for sample in samples}
        groups = {sample.dependency_group for sample in samples}

        self.assertTrue({"gsm8k", "math", "humaneval", "hellaswag", "lambada"} <= tasks)
        self.assertEqual(groups, {"high", "low"})

    def test_curated_suite_can_limit_examples_per_task(self):
        samples = load_curated_benchmark_samples(
            tasks=["gsm8k", "lambada"],
            max_examples_per_task=1,
        )

        self.assertEqual([sample.task for sample in samples], ["gsm8k", "lambada"])

    def test_write_samples_jsonl_uses_cli_schema(self):
        samples = load_curated_benchmark_samples(tasks=["gsm8k"], max_examples_per_task=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bench.jsonl"
            write_samples_jsonl(path, samples)
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["task"], "gsm8k")
        self.assertEqual(record["dependency_group"], "high")
        self.assertIn("prompt", record)
        self.assertIn("completion", record)


if __name__ == "__main__":
    unittest.main()
