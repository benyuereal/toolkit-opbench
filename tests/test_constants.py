"""bench/hw/constants 的单测。

不依赖 GPU。**依赖上游 constants.yaml 存在**——这是刻意的：本模块的
全部意义就是"不许编数"，所以缺文件时应该报错而不是跳过或回落。

    python3 -m unittest discover -s tests
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from bench.hw import constants as C


@unittest.skipUnless(os.path.isfile(C.UPSTREAM), "上游 constants.yaml 未挂载")
class TestConstants(unittest.TestCase):
    def test_three_kinds_resolve(self):
        self.assertEqual(C.get("peak_nominal:fp8")["kind"], "peak_nominal")
        self.assertEqual(C.get("peak_achievable:bf16")["kind"], "peak_achievable")
        self.assertEqual(C.get("impl_measured:fp8")["kind"], "impl_measured")

    def test_peak_returns_float(self):
        v = C.peak("peak_achievable", "bf16")
        self.assertIsInstance(v, float)
        self.assertGreater(v, 0)

    def test_peak_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            C.peak("nominal", "fp8")

    def test_missing_variant_raises(self):
        # fp8 没有 achievable 口径的值：不许拿 nominal 或 impl 顶替
        with self.assertRaises(KeyError):
            C.get("peak_achievable:fp8")

    def test_unknown_id_raises_without_fallback(self):
        with self.assertRaises(KeyError):
            C.get("no_such_constant_xyz")

    def test_literal_in_id_rejected(self):
        with self.assertRaises(KeyError):
            C.get("fp8_peak_nominal_560")

    def test_suffix_lookup(self):
        self.assertEqual(C.get("bf16_gemm_peak_tf")["kind"], "peak_achievable")

    def test_caliber_cannot_be_mixed(self):
        # 上游声明 impl_measured，按 achievable 引用必须炸
        with mock.patch.dict(C.VARIANT, {("peak_achievable", "fp8"): "compute.fp8_impl_measured_tf"}):
            with self.assertRaises(C.ConstantsError):
                C.get("peak_achievable:fp8")

    def test_status_is_exposed(self):
        self.assertEqual(C.get("peak_nominal:fp8")["status"], "unverified")
        self.assertEqual(C.get("peak_achievable:bf16")["status"], "measured")

    def test_l2_is_bytes(self):
        self.assertEqual(C.l2_bytes(), 8 * 1024 * 1024)

    def test_upstream_clock_is_the_documented_one(self):
        # 上游记 1.15 GHz；本机实测工作档 1200 MHz 存在 SCLK_NOTE 里，不覆盖上游值
        self.assertEqual(C.upstream_clock_mhz(), 1150)
        self.assertEqual(C.SCLK_NOTE["work_mhz_observed"], 1200)

    def test_summary_shape(self):
        s = C.summary()
        self.assertEqual(set(s), set(C.KINDS))
        self.assertIn("bf16", s["peak_achievable"])


class TestMissingUpstream(unittest.TestCase):
    def test_missing_file_raises_not_silently(self):
        """缺上游文件时必须报错。静默回落一个默认值就是又编了一个数。"""
        with mock.patch.object(C, "UPSTREAM", "/nonexistent/constants.yaml"):
            C.reload()
            try:
                with self.assertRaises(C.ConstantsError):
                    C.load_all()
            finally:
                C.reload()


if __name__ == "__main__":
    unittest.main()
