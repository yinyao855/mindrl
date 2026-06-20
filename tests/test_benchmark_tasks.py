import json
import tempfile
import unittest
from pathlib import Path

from mindrl_repo.benchmark_tasks import TaskSample, load_jsonl_samples, load_preset_samples


class BenchmarkTasksTest(unittest.TestCase):
    def test_preset_samples_include_high_and_low_dependency_groups(self):
        samples = load_preset_samples(max_examples=10)

        groups = {sample.dependency_group for sample in samples}
        tasks = {sample.task for sample in samples}

        self.assertIn("high", groups)
        self.assertIn("low", groups)
        self.assertIn("gsm8k_smoke", tasks)
        self.assertIn("hellaswag_smoke", tasks)
        self.assertTrue(all(sample.prompt for sample in samples))
        self.assertTrue(all(sample.completion for sample in samples))

    def test_jsonl_loader_accepts_task_prompt_completion_records(self):
        records = [
            {
                "task": "custom_math",
                "prompt": "Question: 1+1?\nAnswer:",
                "completion": " 2",
                "dependency_group": "high",
            },
            {
                "task": "custom_choice",
                "prompt": "Pick the best ending:",
                "completion": " option A",
                "dependency_group": "low",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            samples = load_jsonl_samples(path, max_examples=1)

        self.assertEqual(
            samples,
            [
                TaskSample(
                    task="custom_math",
                    prompt="Question: 1+1?\nAnswer:",
                    completion=" 2",
                    dependency_group="high",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
