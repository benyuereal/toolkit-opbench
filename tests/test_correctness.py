"""bench/check/correctness 的单测。

核心要证明的一件事：**只在极端输入下出错的实现会被压力数值抓到**。
这是"快但算错"的唯一防线，常规形状上看不出来。
"""
from __future__ import annotations

import unittest

from bench.check import correctness as C


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class TestCompare(unittest.TestCase):
    def test_exact(self):
        import torch

        a = torch.randn(16, 16)
        r = C.compare(a, a.clone(), {"rtol": 1e-3, "atol": 1e-3})
        self.assertTrue(r.ok)
        self.assertEqual(r.max_abs, 0.0)

    def test_within_tolerance(self):
        import torch

        e = torch.ones(8, 8)
        a = e + 1e-4
        self.assertTrue(C.compare(a, e, {"rtol": 1e-2, "atol": 1e-3}).ok)

    def test_beyond_tolerance(self):
        import torch

        e = torch.ones(8, 8)
        r = C.compare(e * 1.5, e, {"rtol": 1e-2, "atol": 1e-3})
        self.assertFalse(r.ok)
        self.assertTrue(r.failed_workloads)

    def test_shape_mismatch(self):
        import torch

        r = C.compare(torch.zeros(2, 3), torch.zeros(3, 2), {"rtol": 1e-3, "atol": 1e-3})
        self.assertFalse(r.ok)
        self.assertIn("shape", r.note)

    def test_nan_detected(self):
        import torch

        a = torch.ones(4, 4)
        a[0, 0] = float("nan")
        r = C.compare(a, torch.ones(4, 4), {"rtol": 1e-3, "atol": 1e-3})
        self.assertFalse(r.ok)
        self.assertIn("NaN", r.note)

    def test_tolerance_required(self):
        import torch

        with self.assertRaises(ValueError):
            C.compare(torch.zeros(2), torch.zeros(2), {})
        with self.assertRaises(ValueError):
            C.compare(torch.zeros(2), torch.zeros(2), {"rtol": 1e-3})

    def test_cos_reported(self):
        import torch

        e = torch.randn(64)
        a = -e
        r = C.compare(a, e, {"rtol": 1e-2, "atol": 1e-2})
        self.assertFalse(r.ok)
        self.assertAlmostEqual(r.cos, -1.0, places=5)


class TestPressureInputs(unittest.TestCase):
    def test_large_scales_floats_only(self):
        import torch

        t = {"f": torch.ones(2, 2), "i": torch.ones(2, 2, dtype=torch.int8)}
        out = C.build_pressure_inputs(t, "large")
        self.assertGreater(float(out["f"].abs().max()), 100.0)
        self.assertEqual(out["i"].dtype, torch.int8)
        self.assertTrue(torch.equal(out["i"], t["i"]))

    def test_mixed_sign(self):
        import torch

        t = {"f": torch.ones(1, 8)}
        out = C.build_pressure_inputs(t, "mixed_sign")
        vals = out["f"].flatten().tolist()
        self.assertIn(1.0, vals)
        self.assertIn(-1.0, vals)

    def test_unknown_kind(self):
        import torch

        with self.assertRaises(ValueError):
            C.build_pressure_inputs({"f": torch.ones(1)}, "nope")


@unittest.skipUnless(_has_gpu(), "压力数值测试需要 GPU/CPU 张量运算")
class TestCheckAll(unittest.TestCase):
    def setUp(self):
        from bench.data import definition as defn_mod
        from bench.data import workload as wl_mod

        self.defn = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var", "domain": {"values": [8, 16]}},
                  "N": {"type": "const", "value": 16}},
            inputs={"A": {"shape": ["M", "N"], "dtype": "float32"},
                    "B": {"shape": ["N", "N"], "dtype": "float32"}},
            outputs={"C": {"shape": ["M", "N"], "dtype": "float32"}},
            tolerance={"rtol": 1e-3, "atol": 1e-3},
            reference="reference.py",
        )
        self.workloads = [
            wl_mod.Workload("pub-1", {"M": 8}, {}, "public"),
            wl_mod.Workload("pub-2", {"M": 16}, {}, "public"),
            wl_mod.Workload("hold-1", {"M": 16}, {}, "holdout"),
        ]

    def _ref(self):
        class Ref:
            @staticmethod
            def arun(A, B):
                return A @ B

        return Ref()

    def test_exact_passes(self):
        r = C.check_all(self.defn, self.workloads,
                        lambda t: t["A"] @ t["B"], self._ref())
        self.assertTrue(r.ok, r.note)
        self.assertEqual(r.max_abs, 0.0)

    def test_wrong_fails_on_public_and_holdout(self):
        import torch

        r = C.check_all(self.defn, self.workloads,
                        lambda t: torch.zeros_like(t["A"] @ t["B"]), self._ref())
        self.assertFalse(r.ok)
        self.assertTrue(any("public" in x for x in r.failed_workloads))
        self.assertTrue(any("holdout" in x for x in r.failed_workloads))

    def test_overflow_only_under_pressure(self):
        """常规输入全过、只在压力数值上截断的实现必须被抓到。

        这正是"快但算错"的典型形态：常规形状看起来完美。
        """
        r = C.check_all(self.defn, self.workloads,
                        lambda t: (t["A"] @ t["B"]).clamp(max=1e3), self._ref())
        self.assertFalse(r.ok)
        self.assertTrue(any("pressure" in x for x in r.failed_workloads),
                        r.failed_workloads)
        self.assertFalse(any(x.startswith("public") for x in r.failed_workloads),
                         "常规形状不该失败——失败的是压力数值那一组")

    def test_missing_arun_raises(self):
        from bench.data import workload as wl_mod

        class NoRef:
            pass

        with self.assertRaises(AttributeError):
            C.run_reference(self.defn, self.workloads[0], NoRef())

    def test_run_reference(self):
        out = C.run_reference(self.defn, self.workloads[0], self._ref())
        self.assertEqual(tuple(out.shape), (8, 16))

    def test_no_tolerance(self):
        self.defn.tolerance = None
        r = C.check_all(self.defn, self.workloads,
                        lambda t: t["A"] @ t["B"], self._ref())
        self.assertFalse(r.ok)
        self.assertIn("tolerance", r.note)


if __name__ == "__main__":
    unittest.main()


class TestIntegerExact(unittest.TestCase):
    """整数输出必须精确相等，不能套浮点容差。"""

    @unittest.skipUnless(_has_gpu(), "需要 GPU")
    def test_index_off_by_one_fails(self):
        """索引差 1 是不同元素，必须判失败（不能被 atol 放过）。"""
        import torch
        from bench.data import definition as defn_mod
        from bench.data import workload as wl_mod

        d = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var", "domain": {"values": [8]}},
                  "E": {"type": "const", "value": 16}},
            inputs={"x": {"shape": ["M", "E"], "dtype": "float32"}},
            outputs={"w": {"shape": ["M", "E"], "dtype": "float32"},
                     "i": {"shape": ["M", "E"], "dtype": "int32"}},
            tolerance={"rtol": 1e-2, "atol": 1e-2}, reference="reference.py",
        )
        ws = [wl_mod.Workload("p", {"M": 8}, {}, "public"),
              wl_mod.Workload("h", {"M": 8}, {}, "holdout")]

        class Ref:
            @staticmethod
            def arun(x):
                w = x.float()
                return w, torch.arange(x.shape[1], dtype=torch.int32,
                                       device=x.device).expand(x.shape[0], -1).contiguous()

        # 索引整体 +1：数值差 1，必须失败
        def bad(t):
            w, i = Ref.arun(t["x"])
            return w, i + 1

        r = C.check_all(d, ws, bad, Ref())
        self.assertFalse(r.ok, "索引差 1 必须判失败")
        self.assertIn("整数输出", r.note)

    @unittest.skipUnless(_has_gpu(), "需要 GPU")
    def test_pressure_skipped_for_int_outputs(self):
        """有整数输出的题跳过压力档：饱和输入下 top-k 索引无唯一解。"""
        import torch
        from bench.data import definition as defn_mod
        from bench.data import workload as wl_mod

        d = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var", "domain": {"values": [8]}},
                  "E": {"type": "const", "value": 16}},
            inputs={"x": {"shape": ["M", "E"], "dtype": "float32"}},
            outputs={"i": {"shape": ["M", "E"], "dtype": "int32"}},
            tolerance={"rtol": 1e-2, "atol": 1e-2}, reference="reference.py",
        )
        ws = [wl_mod.Workload("p", {"M": 8}, {}, "public"),
              wl_mod.Workload("h", {"M": 8}, {}, "holdout")]

        class Ref:
            @staticmethod
            def arun(x):
                return torch.argsort(-x, dim=-1).to(torch.int32)

        r = C.check_all(d, ws, lambda t: Ref.arun(t["x"]), Ref())
        # 若没跳过压力档，×1e4 会让并列导致失败
        self.assertTrue(r.ok, r.note)
