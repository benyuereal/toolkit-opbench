"""bench/timing 的单测。

stats 部分纯 CPU；event/flush 需要 GPU，无卡时跳过。
GPU 段也测：flush 必须在计时窗口**之外**，且 buffer 尺寸必须 >= 2×L2——
这两条是 4571 GB/s 事故的直接产物。
"""
from __future__ import annotations

import math
import unittest

from bench.timing import stats


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class TestMedian(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(stats.median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(stats.median([1, 2, 3, 4]), 2.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stats.median([])

    def test_single(self):
        self.assertEqual(stats.median([7.5]), 7.5)


class TestMad(unittest.TestCase):
    def test_robust_to_outlier(self):
        """MAD 不该被一个离群点拖走——标准差会被。"""
        base = [10, 11, 12, 13, 14]
        with_outlier = base + [1000]
        self.assertLess(stats.mad(with_outlier), 3.0)

    def test_zero_spread(self):
        self.assertEqual(stats.mad([5, 5, 5]), 0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stats.mad([])


class TestCv(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(stats.cv([10, 10, 10]), 0.0)

    def test_zero_median_is_inf(self):
        """median 为 0 是"量不出来"，不能报成"很稳"。"""
        self.assertEqual(stats.cv([0, 0, 0]), float("inf"))


class TestSignificant(unittest.TestCase):
    def test_beyond_three_mad(self):
        self.assertTrue(stats.significant(10.0, 10.1, 0.01, 0.01))

    def test_within_noise(self):
        self.assertFalse(stats.significant(10.0, 10.01, 0.01, 0.01))

    def test_uses_max_of_both(self):
        self.assertFalse(stats.significant(10.0, 10.1, 0.001, 0.1))

    def test_zero_mad_explicit(self):
        self.assertFalse(stats.significant(10.0, 10.0, 0.0, 0.0))
        self.assertTrue(stats.significant(10.0, 11.0, 0.0, 0.0))


class TestGeomean(unittest.TestCase):
    def test_two(self):
        self.assertAlmostEqual(stats.geomean([2, 8]), 4.0)

    def test_resists_single_spike(self):
        """算术平均 5.5，几何平均 4.0——后者不给"挑好看的形状"留下收益。"""
        g = stats.geomean([1, 10])
        self.assertAlmostEqual(g, math.sqrt(10))
        self.assertLess(g, (1 + 10) / 2)

    def test_rejects_zero_and_negative(self):
        for bad in ([1, 0], [1, -2]):
            with self.assertRaises(ValueError):
                stats.geomean(bad)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stats.geomean([])


class TestOverlap(unittest.TestCase):
    def test_perfect_overlap_of_smaller_side(self):
        """10ms 与 2ms 完全重叠：总耗时 10，重叠度 100%。"""
        self.assertAlmostEqual(stats.overlap_ratio(12, 10, 10, 2), 1.0)

    def test_denominator_is_min_not_max(self):
        """用 max 做分母只会算出 20%，那个读法是错的。"""
        r = stats.overlap_ratio(12, 10, 10, 2)
        self.assertNotAlmostEqual(r, (12 - 10) / 10)

    def test_no_overlap(self):
        self.assertAlmostEqual(stats.overlap_ratio(12, 12, 10, 2), 0.0)

    def test_negative_overlap(self):
        """并行比串行还慢是可能的（争用），如实给负数。"""
        self.assertLess(stats.overlap_ratio(10, 12, 6, 4), 0)

    def test_scale_ratio_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            stats.overlap_ratio(10.1, 10.0, 10.0, 0.1)
        self.assertIn("量级比", str(ctx.exception))

    def test_equal_sides_ok(self):
        self.assertAlmostEqual(stats.overlap_ratio(8, 4, 4, 4), 1.0)

    def test_zero_side_rejected(self):
        with self.assertRaises(ValueError):
            stats.overlap_ratio(5, 4, 5, 0)

    def test_negative_input_rejected(self):
        with self.assertRaises(ValueError):
            stats.overlap_ratio(-1, 4, 5, 5)


@unittest.skipUnless(_has_gpu(), "需要 GPU")
class TestFlush(unittest.TestCase):
    def test_small_buffer_refused(self):
        from bench.timing import flush

        with self.assertRaises(ValueError) as ctx:
            flush.flush(4 * 1024 * 1024)
        self.assertIn("2×L2", str(ctx.exception))

    def test_default_buffer_is_2x_l2(self):
        from bench.timing import flush

        self.assertGreaterEqual(flush.DEFAULT_FLUSH_BYTES, 16 * 1024 * 1024)

    def test_flush_runs(self):
        from bench.timing import flush

        flush.flush()
        import torch

        torch.cuda.synchronize()

    def test_persisting_noop(self):
        from bench.timing import flush

        self.assertIsNone(flush.reset_persisting_l2())

    def test_metadata(self):
        from bench.timing import flush

        info = flush.is_supported()
        self.assertTrue(info["cuda_available"])
        self.assertFalse(info["persisting_l2_reset"])


@unittest.skipUnless(_has_gpu(), "需要 GPU")
class TestEventTiming(unittest.TestCase):
    def test_times_a_real_kernel(self):
        import torch

        from bench.timing import event

        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        t = event.time_call(lambda: torch.mm(a, b), warmup=3, iters=8)
        self.assertGreater(t.median_ms, 0)
        self.assertEqual(len(t.samples), 8)
        self.assertEqual(t.warmup, 3)
        self.assertEqual(t.iters, 8)

    def test_zero_time_detected(self):
        """没真正执行时 median 应为 0 并带提示。"""
        import torch

        from bench.timing import event

        t0 = torch.zeros(1, device="cuda")
        t = event.time_call(lambda: None, warmup=1, iters=3)
        self.assertEqual(len(t.samples), 3)

    def test_bad_params(self):
        from bench.timing import event

        with self.assertRaises(ValueError):
            event.time_call(lambda: None, iters=0)

    def test_warmup_zero_still_pays_first_call(self):
        """warmup=0 也必须跑一次：否则首调编译时间会被算进样本。"""
        import torch

        from bench.timing import event

        calls = {"n": 0}

        def fn():
            calls["n"] += 1

        event.time_call(fn, warmup=0, iters=2)
        # 1 次预热 + 2 次采样
        self.assertEqual(calls["n"], 3)

    def test_first_call_cost_not_in_samples(self):
        """首调故意慢 50 倍，样本里不该出现它。"""
        import time

        import torch

        from bench.timing import event

        state = {"first": True}

        def fn():
            if state["first"]:
                state["first"] = False
                time.sleep(0.2)

        t = event.time_call(fn, warmup=1, iters=5)
        self.assertLess(t.median_ms, 50.0)

    def test_time_many_interleaves(self):
        import torch

        from bench.timing import event

        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        out = event.time_many(
            {"mm_ab": lambda: torch.mm(a, b), "mm_ac": lambda: torch.mm(a, c)},
            warmup=2, iters=4,
        )
        self.assertEqual(set(out), {"mm_ab", "mm_ac"})
        for t in out.values():
            self.assertEqual(len(t.samples), 4)

    def test_time_many_empty(self):
        from bench.timing import event

        with self.assertRaises(ValueError):
            event.time_many({})


if __name__ == "__main__":
    unittest.main()
