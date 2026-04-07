from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from torch_probe import run_backend_probe, run_threadpool_probe


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TorchProbeTests(unittest.TestCase):
    def test_direct_threadpool_probe_runs(self) -> None:
        result = run_threadpool_probe(job_count=4, max_workers=2)

        self.assertEqual(result["job_count"], 4)
        self.assertEqual(result["max_workers"], 2)
        self.assertEqual(len(result["sample_output"]), 2)

    def test_backend_probe_runs_torch_in_executor(self) -> None:
        result = run_backend_probe()

        self.assertEqual(result["result_count"], 2)
        self.assertFalse(result["errors"])
        self.assertEqual(result["payload"]["sample_rate"], 16_000)
        self.assertEqual(len(result["payload"]["scores"]), 2)


if __name__ == "__main__":
    unittest.main()
