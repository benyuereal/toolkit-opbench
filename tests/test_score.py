"""bench/score 四个模块的单测。

不依赖 GPU、不依赖 pytest：`python3 -m unittest discover -s tests` 即可。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bench.score import calibrate, judgment, ratio as ratio_mod, sol_like


class TestRatio(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(ratio_mod.ratio(10.0, 5.0), 2.0)

    def test_slower_than_baseline(self):
        self.assertAlmostEqual(ratio_mod.ratio(10.0, 20.0), 0.5)

    def test_rejects_zero_and_negative(self):
        with self.assertRaises(ValueError):
            ratio_mod.ratio(0.0, 1.0)
        with self.assertRaises(ValueError):
            ratio_mod.ratio(1.0, -2.0)

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            ratio_mod.ratio(float("nan"), 1.0)

    def test_geomean(self):
        # sqrt(2 * 8) = 4；算术平均会是 5，几何平均能压住单个超大值
        self.assertAlmostEqual(
            ratio_mod.geomean_ratio([(2.0, 1.0), (8.0, 1.0)]), 4.0
        )

    def test_geomean_empty(self):
        with self.assertRaises(ValueError):
            ratio_mod.geomean_ratio([])

    def test_load_baseline_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ratio_mod.load_baseline("nope", d)

    def test_load_baseline_ok(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t1.json"), "w", encoding="utf-8") as f:
                json.dump({"t_b_ms": 10.0, "t_sol_ms": 2.0}, f)
            data = ratio_mod.load_baseline("t1", d)
            self.assertEqual(data["t_b_ms"], 10.0)

    def test_load_baseline_bad_field(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t2.json"), "w", encoding="utf-8") as f:
                json.dump({"t_b_ms": "10", "t_sol_ms": 2.0}, f)
            with self.assertRaises(TypeError):
                ratio_mod.load_baseline("t2", d)


class TestSolLike(unittest.TestCase):
    def test_baseline_equals_half(self):
        # T_k == T_b 时正好 0.5
        self.assertAlmostEqual(sol_like.score(10.0, 10.0, 2.0), 0.5)

    def test_speed_of_light_equals_one(self):
        self.assertAlmostEqual(sol_like.score(2.0, 10.0, 2.0), 1.0)

    def test_degenerate_baseline_equals_sol(self):
        # 基线与光速相同：分母为 0，按物理意义判定
        self.assertEqual(sol_like.score(5.0, 10.0, 10.0), 1.0)
        self.assertEqual(sol_like.score(11.0, 10.0, 10.0), 0.0)

    def test_faster_than_light_is_flagged(self):
        s = sol_like.score(1.0, 10.0, 2.0)
        self.assertGreater(s, 1.0)
        self.assertTrue(sol_like.is_suspicious(s))

    def test_from_ratio_matches_direct(self):
        direct = sol_like.score(5.0, 10.0, 2.0)
        self.assertAlmostEqual(sol_like.from_ratio(2.0, 10.0, 2.0), direct)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            sol_like.score(0.0, 10.0, 2.0)


class TestJudgment(unittest.TestCase):
    def setUp(self):
        self.defn = {
            "expected_verdict": "不可用",
            "verdict_keywords": ["不可用", "被拒", "不支持"],
            "forbidden_claims": ["可以融合", "有 3 倍空间"],
            "evidence_required": ["error_output", "command"],
        }

    def test_full_score(self):
        """满分要求：方向对 + 证据种类齐 + 有定量结果。"""
        s, why = judgment.score(
            "asm 后端在 gfx938 上不可用",
            ["error_output: RuntimeError: unsupport current input type",
             "command: python3 scripts/probe.py --dtype bf16",
             "实测 rel_l2 1.66e-3"],
            self.defn,
        )
        self.assertEqual(s, 100, why)

    def test_qualified_without_quantitative(self):
        """方向对、证据种类齐，但没有定量结果 → 80，不是 100。"""
        s, why = judgment.score(
            "asm 后端在 gfx938 上不可用",
            ["error_output: RuntimeError: unsupport current input type",
             "command: python3 scripts/probe.py --dtype bf16"],
            self.defn,
        )
        self.assertEqual(s, 80, why)
        self.assertIn("定量", why)

    def test_wrong_direction_scores_zero(self):
        """方向答反：证据再齐全也是 0。这是"答错也得 60"那次的直接防线。"""
        s, why = judgment.score(
            "asm 后端可用",
            ["error_output: RuntimeError: boom",
             "command: python3 probe.py",
             "rel_l2 1e-3"],
            self.defn,
        )
        self.assertEqual(s, 0, why)
        self.assertIn("方向错", why)

    def test_self_admitted_unverified_scores_80(self):
        """有数但自承没验成 → 80。诚实但不等于验成了。"""
        s, why = judgment.score(
            "asm 后端不可用。但本次未能在数值上复核。",
            ["error_output: RuntimeError: boom",
             "command: python3 probe.py",
             "cos≈0.10"],
            self.defn,
        )
        self.assertEqual(s, 80, why)
        self.assertIn("未完成数值验证", why)

    def test_thin_evidence(self):
        s, why = judgment.score("asm 不可用", ["只有一条"], self.defn)
        self.assertEqual(s, 60, why)

    def test_evidence_kinds_all_required(self):
        """两条证据但都只是"我测了"，缺 command/error_output 种类 → 60 分。"""
        s, why = judgment.score("asm 不可用", ["我测了两次", "都没成"], self.defn)
        self.assertEqual(s, 60, why)
        self.assertIn("种类", why)

    def test_integer_evidence_required_still_supported(self):
        d = dict(self.defn, evidence_required=2)
        s, _ = judgment.score("asm 不可用", ["a", "b", "rel_l2 1e-3"], d)
        self.assertEqual(s, 100)

    def test_honest_uncertainty(self):
        s, why = judgment.score("还没测，未测定", [], self.defn)
        self.assertEqual(s, 20, why)

    def test_forbidden_claim_zero(self):
        s, why = judgment.score("其实可以融合，比拆分快", ["测过了"], self.defn)
        self.assertEqual(s, 0, why)
        self.assertIn("禁止", why)

    def test_empty_verdict_zero(self):
        s, _ = judgment.score("", ["x"], self.defn)
        self.assertEqual(s, 0)

    def test_waffle_without_direction_zero(self):
        s, why = judgment.score("需要具体情况具体分析", ["x", "y"], self.defn)
        self.assertEqual(s, 0, why)

    def test_is_forbidden_case_insensitive_ascii(self):
        hits = judgment.is_forbidden("ASM backend is UNSUPPORTED", ["unsupported"])
        self.assertEqual(len(hits), 1)


class TestCalibrate(unittest.TestCase):
    def test_extract_claim(self):
        text = "该 kernel 已达 92% 峰值带宽，误差 1.2%。"
        self.assertAlmostEqual(
            calibrate.claimed_achievement(text, peak=500.0, measured=460.0), 92.0
        )

    def test_extract_prefers_achievement_sentence(self):
        text = "占比 30%。达成率 61%。"
        self.assertAlmostEqual(
            calibrate.claimed_achievement(text, peak=500.0, measured=305.0), 61.0
        )

    def test_extract_none(self):
        self.assertIsNone(calibrate.claimed_achievement("没有数字", 500.0, 100.0))

    def test_error_and_pass(self):
        self.assertAlmostEqual(calibrate.calibration_error(92.0, 61.0, peak=None), 31.0)
        self.assertFalse(calibrate.is_calibrated(92.0, 61.0))
        self.assertTrue(calibrate.is_calibrated(64.0, 61.0))

    def test_error_with_peak(self):
        # measured=305 / peak=500 = 61%
        self.assertAlmostEqual(calibrate.calibration_error(70.0, 305.0, peak=500.0), 9.0)
        self.assertTrue(calibrate.is_calibrated(70.0, 305.0, peak=500.0))

    def test_unknown_peak_raises(self):
        with self.assertRaises(ValueError):
            calibrate.calibration_error(50.0, 100.0, peak=0.0)

    def test_achievement_pct_guards(self):
        self.assertIsNone(calibrate.achievement_pct(100.0, 0.0))
        self.assertAlmostEqual(calibrate.achievement_pct(250.0, 500.0), 50.0)


if __name__ == "__main__":
    unittest.main()
