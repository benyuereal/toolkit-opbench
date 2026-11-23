"""CLI 入口的退出码契约。

这条测试的存在理由：`make verify` / `make score` 曾经是静默空操作——
模块没有 __main__，`python3 -m ...` 成功退出、什么都没检查，CI 一片绿。
"计数器给了、闸门没关"已经发生三次，必须用测试钉住。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT,
                          capture_output=True, text=True)


class TestVerifyCli(unittest.TestCase):
    def test_exists_and_accepts_run(self):
        r = run("bench.check.integrity", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--run", r.stdout)

    def test_missing_dir_nonzero(self):
        r = run("bench.check.integrity", "--run", "/nope/nope")
        self.assertNotEqual(r.returncode, 0, "目录不存在必须非 0")

    def test_bad_status_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t1", "attempt-1")
            os.makedirs(p)
            with open(os.path.join(p, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "error"}, f)
            r = run("bench.check.integrity", "--run", d)
            self.assertNotEqual(r.returncode, 0, "status!=ok 必须非 0")

    def test_clean_run_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t1", "attempt-1")
            os.makedirs(p)
            with open(os.path.join(p, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "ok"}, f)
            with open(os.path.join(p, "_stdout.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"type":"result"}\n')
            r = run("bench.check.integrity", "--run", d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_answer_leak_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t1", "attempt-1")
            os.makedirs(p)
            with open(os.path.join(p, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "ok"}, f)
            with open(os.path.join(p, "_stdout.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"type":"result"}\n')
            with open(os.path.join(p, "report.md"), "w", encoding="utf-8") as f:
                f.write("我看了 expect.md 的 expected_verdict")
            r = run("bench.check.integrity", "--run", d)
            self.assertNotEqual(r.returncode, 0, "答案泄露必须非 0")


class TestScoreCli(unittest.TestCase):
    def test_exists(self):
        r = run("bench.score.ratio", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--run", r.stdout)

    def test_missing_baseline_nonzero(self):
        """缺基线必须报错，不许当成"没东西可算"就通过。"""
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as bd:
            p = os.path.join(rd, "t1", "attempt-1", "results")
            os.makedirs(p)
            with open(os.path.join(p, "r.json"), "w", encoding="utf-8") as f:
                json.dump({"timing": {"median_ms": 1.0}}, f)
            r = run("bench.score.ratio", "--run", rd, "--baselines", bd)
            self.assertNotEqual(r.returncode, 0, "缺基线必须非 0")

    def test_computes_ratio(self):
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as bd:
            p = os.path.join(rd, "t1", "attempt-1", "results")
            os.makedirs(p)
            with open(os.path.join(p, "r.json"), "w", encoding="utf-8") as f:
                json.dump({"timing": {"median_ms": 5.0}}, f)
            with open(os.path.join(bd, "t1.json"), "w", encoding="utf-8") as f:
                json.dump({"t_b_ms": 10.0, "t_sol_ms": 2.0}, f)
            r = run("bench.score.ratio", "--run", rd, "--baselines", bd, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(r.stdout)
            self.assertAlmostEqual(data["rows"]["t1"]["ratio"], 2.0)


class TestMakeTargets(unittest.TestCase):
    """make 的目标必须真的接上入口，不能是空壳。"""

    def test_make_verify_not_noop(self):
        r = subprocess.run(["make", "verify", "ARM=A"], cwd=ROOT,
                           capture_output=True, text=True)
        # arm-A 的 status 是 error，所以必须非 0
        self.assertNotEqual(r.returncode, 0,
                            "make verify 对 status=error 的运行必须失败")

    def test_make_score_reports_missing_baseline(self):
        r = subprocess.run(["make", "score", "ARM=A"], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0,
                            "make score 缺基线时必须失败，不能静默通过")


if __name__ == "__main__":
    unittest.main()
