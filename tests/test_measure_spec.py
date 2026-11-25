"""测量口径守卫：三臂 prompt 必须逐字相同，且都必须带上口径。

为什么要有这个测试：
    2026-11-24 实测发现 A 臂自选 SKV=16384/TOPK=2048，C/D 用 8192/256，
    S=512 时 2542us vs 163us —— 差 15 倍。那不是优化差异，是测的不是同一个算子。
    口径现在由 harness 下发，这个测试防止有人把注入逻辑删掉、或者改出臂间差异。
"""
import unittest

from harness import measure_spec
from harness.arms import a, c, skillhub


class TestMeasureSpec(unittest.TestCase):
    TASK = "mla-sparse-l1"

    def test_spec_contains_all_baseline_uuids(self):
        import json, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = json.load(open(os.path.join(root, "baselines", f"{self.TASK}.json"), encoding="utf-8"))
        spec = measure_spec.render(self.TASK)
        for s in base.get("shapes") or []:
            self.assertIn(str(s.get("uuid")), spec)

    def test_spec_requires_bench_std(self):
        spec = measure_spec.render(self.TASK)
        self.assertIn("bench_std.json", spec)
        self.assertIn("测量口径", spec)

    def test_three_arms_prompt_identical(self):
        task_md = "# 题面\n内容\n"
        spec = measure_spec.render(self.TASK)
        p_a = a.build_prompt(task_md).rstrip() + spec
        p_c = c.build_prompt(task_md).rstrip() + spec
        p_d = skillhub.build_prompt(task_md).rstrip() + spec
        self.assertEqual(p_a, p_c, "A 与 C 的 prompt 必须逐字相同")
        self.assertEqual(p_c, p_d, "C 与 D 的 prompt 必须逐字相同")

    def test_unknown_task_returns_empty(self):
        self.assertEqual(measure_spec.render("no-such-task-xyz"), "")


class TestShapeCheck(unittest.TestCase):
    def _load(self):
        import importlib.util, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "report_std", os.path.join(root, "tools", "report_std.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_wrong_shape_is_rejected(self):
        mod = self._load()
        base = {"shapes": [{"uuid": "pub-S512", "axes": {"H": 16, "SKV": 8192, "TOPK": 256}}]}
        bad = {"shapes": [{"uuid": "pub-S512", "axes": {"H": 128, "SKV": 16384, "TOPK": 2048}}]}
        ok, msg = mod.check_shapes(bad, base)
        self.assertFalse(ok)
        self.assertIn("不一致", msg)

    def test_correct_shape_passes(self):
        mod = self._load()
        base = {"shapes": [{"uuid": "pub-S512", "axes": {"H": 16, "SKV": 8192, "TOPK": 256}}]}
        good = {"shapes": [{"uuid": "pub-S512", "axes": {"H": 16, "SKV": 8192, "TOPK": 256},
                            "median_ms": 0.163}]}
        ok, msg = mod.check_shapes(good, base)
        self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
