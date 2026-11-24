"""统一口径实测工具的单测。"""
from __future__ import annotations

import unittest

from harness import measure


class TestResolve(unittest.TestCase):
    def test_resolves_builtin(self):
        self.assertTrue(callable(measure.resolve("json.dumps")))

    def test_missing_attr(self):
        with self.assertRaises(AttributeError):
            measure.resolve("json.no_such")

    def test_not_dotted(self):
        with self.assertRaises(ValueError):
            measure.resolve("dumps")


class TestPerCall(unittest.TestCase):
    @unittest.skipUnless(__import__("torch").cuda.is_available(), "需要 GPU")
    def test_returns_median_mad(self):
        import torch

        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        med, mad, xs = measure.per_call_ms(lambda: a @ a, iters=8, warmup=2)
        self.assertGreater(med, 0)
        self.assertGreaterEqual(mad, 0)
        self.assertEqual(len(xs), 8)


if __name__ == "__main__":
    unittest.main()
