"""harness/collect 的单测：只读产物、三臂配对、胜负线。

用临时目录造 runs/，不碰真实运行结果——真实结果会变，测试必须稳定。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from harness import collect as C


def _mkrun(root: str, arm: str, task: str, *, report: str | None,
           results: dict | None = None, status: str = "ok") -> str:
    d = os.path.join(root, f"arm-{arm}", task, "attempt-1")
    os.makedirs(d, exist_ok=True)
    meta = {"task": task, "arm": arm, "status": status, "wall_time_s": 12.0,
            "tokens": {"output_tokens": 100}, "artifacts": []}
    arts = []
    if report is not None:
        with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        arts.append("report.md")
    if results:
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        for name, obj in results.items():
            with open(os.path.join(d, "results", name), "w", encoding="utf-8") as f:
                json.dump(obj, f)
            arts.append(f"results/{name}")
    meta["artifacts"] = arts
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return d


GOOD = """# 结论

不可用。asm_fmoe_stage1 对 bfloat16 报错。

## 证据

报错原文：RuntimeError: unsupport current input type bfloat16

复跑命令：python3 scripts/probe.py --dtype bf16

import aiter 后 hasattr 为 True，调用被拒。
"""

# kernel 类的"有效结论"：命令 + 带单位的数字
GOOD_KERNEL = """# 结论

落到 aiter.gemm_a16w16，实测 1.23 ms。

复跑：python3 scripts/bench.py
"""


class TestEffectiveConclusion(unittest.TestCase):
    def test_command_and_number(self):
        ok, _ = C.has_effective_conclusion("跑 python3 x.py 得到 1.23 ms", [])
        self.assertTrue(ok)

    def test_judgment_kind_does_not_need_numbers(self):
        """L5 的正确答案是"通不通"，本来就没有性能数字。"""
        ok, why = C.has_effective_conclusion("不可用\n复跑：python3 x.py", [],
                                             kind="judgment")
        self.assertTrue(ok, why)
        ok, why = C.has_effective_conclusion("不可用", [], kind="judgment")
        self.assertFalse(ok, why)

    def test_kernel_kind_needs_numbers(self):
        ok, _ = C.has_effective_conclusion("复跑 python3 x.py", [], kind="kernel")
        self.assertFalse(ok)

    def test_number_without_method(self):
        ok, why = C.has_effective_conclusion("结果是 1.23 ms", [])
        self.assertFalse(ok)
        self.assertIn("复跑", why)

    def test_command_with_raw_output(self):
        ok, _ = C.has_effective_conclusion("python3 x.py", ["results/out.json"])
        self.assertTrue(ok)

    def test_empty_report(self):
        ok, why = C.has_effective_conclusion("", [])
        self.assertFalse(ok)
        self.assertIn("report.md", why)

    def test_no_numbers(self):
        ok, why = C.has_effective_conclusion("python3 x.py\n结论：不可用", [])
        self.assertFalse(ok)
        self.assertIn("数字", why)


class TestCollect(unittest.TestCase):
    def test_pairs_all_three_arms(self):
        with tempfile.TemporaryDirectory() as d:
            for arm in "ABC":
                _mkrun(d, arm, "t1", report=GOOD, results={"r.json": {"x": 1}})
            data = C.collect(d)
            self.assertEqual(data["tasks"], ["t1"])
            self.assertTrue(data["deltas"]["balanced"])
            for arm in "ABC":
                self.assertIn(("t1", arm), data["table"])

    def test_unbalanced_marks_reference_only(self):
        with tempfile.TemporaryDirectory() as d:
            for arm in "AC":
                _mkrun(d, arm, "t1", report=GOOD)
            data = C.collect(d)
            self.assertFalse(data["deltas"]["balanced"])
            self.assertTrue(any("题数不一致" in v for v in data["deltas"]["verdicts"]))

    def test_only_attempt_1_counts(self):
        """attempt-2 是补跑，不参与配对——补跑破坏固定预算的可比性。"""
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "t1", report=GOOD)
            p = os.path.join(d, "arm-A", "t1", "attempt-2")
            os.makedirs(p)
            with open(os.path.join(p, "report.md"), "w", encoding="utf-8") as f:
                f.write(GOOD)
            data = C.collect(d)
            self.assertEqual(data["deltas"]["A"]["n"], 1)

    def test_leak_marks_row(self):
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "t1", report="我用了 fuse_groupgemm_permute_quant")
            data = C.collect(d)
            self.assertTrue(data["table"][("t1", "A")]["leaked"])

    def test_judgment_uses_score_module(self):
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "moe-asm-fmoe-l5", report=GOOD)
            data = C.collect(d)
            row = data["table"][("moe-asm-fmoe-l5", "A")]
            self.assertIsNotNone(row["judged"])
            self.assertGreaterEqual(row["judged"], 60)

    def test_forbidden_claim_scores_zero(self):
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "moe-asm-fmoe-l5",
                   report="可以融合，预期提升 20%\n\npython3 x.py\n")
            data = C.collect(d)
            self.assertEqual(data["table"][("moe-asm-fmoe-l5", "A")]["judged"], 0)


class TestDeltas(unittest.TestCase):
    def test_winner_lines_fire(self):
        with tempfile.TemporaryDirectory() as d:
            # A 无有效结论，B/C 有：Δ_eff(C−A)=100pt 不该触发"<20pt 砍文档"，
            # 但 B 与 C 相同 ⇒ 该触发"判据层无价值"
            _mkrun(d, "A", "t1", report="不可用")
            _mkrun(d, "B", "t1", report=GOOD_KERNEL)
            _mkrun(d, "C", "t1", report=GOOD_KERNEL)
            data = C.collect(d)
            v = data["deltas"]["verdicts"]
            self.assertEqual(data["deltas"]["d_eff_C_minus_A"], 100.0)
            self.assertTrue(any("B 与 C" in x for x in v), v)
            self.assertFalse(any("砍文档" in x for x in v), v)

    def test_no_gain_line_fires(self):
        """三臂一样好 ⇒ 不触发"砍文档"那条（Δ=0 也是 <20，但语义是没增益）。"""
        with tempfile.TemporaryDirectory() as d:
            for arm in "ABC":
                _mkrun(d, arm, "t1", report=GOOD_KERNEL)
            v = C.collect(d)["deltas"]["verdicts"]
            self.assertTrue(any("砍文档" in x for x in v), v)

    def test_hallucination_regression_fires(self):
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "t1", report="python3 x.py 得到 1 ms")
            _mkrun(d, "B", "t1", report="python3 x.py 得到 1 ms")
            _mkrun(d, "C", "t1",
                   report="python3 x.py 得到 1 ms\n调用 ghostlib.no_such_fn(3) 加速 9 ms")
            data = C.collect(d)
            v = data["deltas"]["verdicts"]
            self.assertTrue(any("立刻停" in x for x in v), v)


class TestRender(unittest.TestCase):
    def test_report_has_all_sections(self):
        with tempfile.TemporaryDirectory() as d:
            for arm in "ABC":
                _mkrun(d, arm, "t1", report=GOOD_KERNEL)
            text = C.render_report(C.collect(d))
        for section in ("## 结论一句话", "## 三臂对比", "## Δ", "## 泄漏与污染"):
            self.assertIn(section, text)

    def test_missing_arm_shows_unrun(self):
        with tempfile.TemporaryDirectory() as d:
            _mkrun(d, "A", "t1", report=GOOD)
            text = C.render_report(C.collect(d))
        self.assertIn("未跑", text)

    def test_write_report(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
            _mkrun(d, "A", "t1", report=GOOD)
            path = C.write_report(d, out)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.path.isfile(os.path.join(out, "report.json")))


if __name__ == "__main__":
    unittest.main()
