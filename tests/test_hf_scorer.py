import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local test env
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class HFScorerTest(unittest.TestCase):
    def test_resolve_device_cpu(self):
        from mindrl_repo.hf_scorer import resolve_device

        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_resolve_device_auto_returns_torch_device(self):
        from mindrl_repo.hf_scorer import resolve_device

        self.assertIsInstance(resolve_device("auto"), torch.device)


if __name__ == "__main__":
    unittest.main()
