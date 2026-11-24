"""harness.ready 自检工具的单测。"""
from __future__ import annotations

import unittest

import harness.ready as R


class TestReady(unittest.TestCase):
    def test_schema_ok(self):
        r = R.check_task_schema()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_complete_ok(self):
        r = R.check_task_complete()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_answer_isolated(self):
        r = R.check_answer_isolated()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_baseline_ok(self):
        r = R.check_baseline()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_judge_scoring_ok(self):
        r = R.check_judge_scoring()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_tooling_ok(self):
        r = R.check_tooling()
        self.assertEqual(r["level"], "ok", r["value"])

    def test_render_counts(self):
        table = {n: {"level": "ok", "value": "v", "ok": True, "fix_hint": ""}
                 for n in R.CHECKS}
        out = R.render(table)
        self.assertIn("可以开跑", out)
        table["baseline"] = {"level": "fail", "value": "x", "ok": False,
                             "fix_hint": "修一下"}
        self.assertIn("不能跑", R.render(table))

    def test_exit_code(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIn(R.main([]), (0, 1))


if __name__ == "__main__":
    unittest.main()
