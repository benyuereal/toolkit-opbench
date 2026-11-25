"""多臂模块的单测。锁住的是"历史事故不许复发"，不是实现细节。"""
import json
import os
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCaliber(unittest.TestCase):
    def test_same_task_shares_one_caliber(self):
        """同一题的各臂必须拿到**同一个**口径实例。

        历史事故：各臂自选形状（SKV 16384 vs 8192），同一"算子"差 15 倍。
        """
        from harness.multi.model import build_units

        units = build_units([{"id": "norm-fused-add-rmsnorm-l1", "budget": 3600}],
                            ["A", "C", "D"])
        self.assertEqual(len(units), 3)
        calibers = {u.caliber for u in units}
        self.assertEqual(len(calibers), 1, "同一题出现了多套口径")
        digests = {u.caliber.digest() for u in units}
        self.assertEqual(len(digests), 1)

    def test_caliber_from_baseline_only(self):
        """口径只能来自 baselines/*.json，不许在代码里写字面量。"""
        from harness.multi.model import from_baseline

        cal = from_baseline("norm-fused-add-rmsnorm-l1")
        self.assertIsNotNone(cal)
        with open(os.path.join(ROOT, "baselines", "norm-fused-add-rmsnorm-l1.json")) as f:
            b = json.load(f)
        self.assertAlmostEqual(cal.t_b_ms, b["t_b_ms"])
        self.assertAlmostEqual(cal.t_sol_ms, b["t_sol_ms"])
        self.assertEqual(cal.peak_id, b["peak_id"])

    def test_judgment_task_has_no_caliber(self):
        """judgment 题没有性能基线，不构造口径（也不是错误）。"""
        from harness.multi.model import from_baseline

        self.assertIsNone(from_baseline("judge-ghost-libs-l3"))

    def test_calibration_mismatch_detected(self):
        """两道不同题的口径若被串到一起，必须能发现。"""
        from harness.multi.model import Unit, check_calibration, from_baseline

        u1 = Unit("norm-fused-add-rmsnorm-l1", "A", 100, caliber=from_baseline("norm-fused-add-rmsnorm-l1"))
        u2 = Unit("gemm-bf16-l1", "A", 100, caliber=from_baseline("gemm-bf16-l1"))
        self.assertEqual(check_calibration([u1, u2]), [])

        # 手工造一个串了口径的单元
        u3 = Unit("norm-fused-add-rmsnorm-l1", "C", 100, caliber=from_baseline("gemm-bf16-l1"))
        problems = check_calibration([u1, u3])
        self.assertTrue(problems, "同题两套口径没被发现")


class TestSuite(unittest.TestCase):
    def test_parse_inline_map(self):
        from harness.multi.suite import load_suite

        tasks = load_suite("three-way")
        self.assertEqual([t["id"] for t in tasks][:1], ["moe-gating-l1"])
        self.assertTrue(all(t["budget"] for t in tasks))

    def test_parse_multiline_yaml(self):
        from harness.multi.suite import load_suite

        tasks = load_suite("l5-only")
        ids = [t["id"] for t in tasks]
        self.assertIn("gemm-fp8-l5", ids)
        # budget_per_task: 600 要作为默认预算传下去
        self.assertTrue(all(t["budget"] == 600 for t in tasks))


class TestIsolation(unittest.TestCase):
    def test_arms_have_distinct_users(self):
        """一臂一用户是硬要求。同 uid 下 700 不生效。"""
        from harness.multi import isolation as iso

        users = [iso.arm_user(a) for a in ("A", "C", "D")]
        self.assertTrue(all(users), f"有臂没有独立用户：{users}")
        self.assertEqual(len(set(users)), 3, f"有臂共用用户：{users}")

    def test_no_shared_user_allowed(self):
        problems = []

        class FakeIso:
            pass

        # 直接验证逻辑：同一 user 出现两次必须报错
        import harness.multi.isolation as iso

        orig = iso.arm_user
        try:
            iso.arm_user = lambda a: "same_user"
            problems = iso.ensure_users(["A", "C"])
        finally:
            iso.arm_user = orig
        self.assertTrue(any("共用用户" in p for p in problems), problems)

    def test_legacy_scan_flags_world_readable(self):
        from harness.multi import isolation as iso

        recs = iso.scan_legacy()
        # 只要求结构正确：有 world_readable 字段，且能识别 755
        for r in recs:
            self.assertIn("world_readable", r)
            if r["mode"] in ("0o755", "0o777"):
                self.assertTrue(r["world_readable"])


class TestPreflight(unittest.TestCase):
    def test_budget_below_stated_is_blocked(self):
        """压预算必须被拦。历史事故：--budget 120 对 1800s 的题，产出全是 timeout。"""
        from harness.multi import preflight
        from harness.multi.model import build_units

        units = build_units([{"id": "norm-fused-add-rmsnorm-l1", "budget": 60}],
                            ["A"])
        problems = preflight.check_budgets(units)
        self.assertTrue(problems, "低于题面预算没被拦")
        self.assertIn("budget", problems[0])

    def test_missing_env_reported(self):
        from harness.multi import preflight

        problems, _info = preflight.check_env("/nonexistent/env/file")
        self.assertTrue(problems)
        self.assertIn("不存在", problems[0])

    def test_trailing_slash_url_handled(self):
        """ANTHROPIC_BASE_URL 尾斜杠必须被处理掉（否则 //v1/messages 得 404）。

        `_load_env()` 会 rstrip，所以正确处理的结果是：**不是 problem**，
        但要留一条 info 说明源文件有斜杠、已被修正。若哪天 rstrip 被去掉，
        这里会变成 problem。
        """
        from harness.multi import preflight
        from harness.run import _load_env

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("ANTHROPIC_AUTH_TOKEN=sk-test\n")
            f.write("ANTHROPIC_BASE_URL=http://10.0.0.1:4001/\n")
            path = f.name
        try:
            problems, info = preflight.check_env(path)
            # 传给子进程的值必须没有尾斜杠
            self.assertFalse(_load_env()["ANTHROPIC_BASE_URL"].endswith("/"))
            self.assertFalse([p for p in problems if "尾斜杠" in p],
                             f"rstrip 失效会让三臂 404：{problems}")
            self.assertTrue(any("尾斜杠" in i for i in info), info)
        finally:
            os.unlink(path)


class TestProgress(unittest.TestCase):
    def test_done_requires_artifact_not_meta(self):
        """meta.json 只说明跑过，不能当"有产物"。"""
        from harness.multi.progress import Snapshot

        s = Snapshot("t", "A", "/tmp/x", has_meta=True, has_stdout=True,
                     has_report=False, n_results=0, n_artifacts=3,
                     last_mtime=None, status="ok", wall_s=1.0)
        self.assertFalse(s.done)

    def test_stalled_when_old_and_no_artifacts(self):
        import time as _t

        from harness.multi.progress import STALL_S, Snapshot

        s = Snapshot("t", "A", "/tmp/x", True, False, False, 0, 0,
                     last_mtime=_t.time() - STALL_S - 60, status="running", wall_s=None)
        self.assertTrue(s.stalled)
        s2 = Snapshot("t", "A", "/tmp/x", True, False, True, 0, 0,
                      last_mtime=_t.time() - STALL_S - 60, status="running", wall_s=None)
        self.assertFalse(s2.stalled, "有 report 的不该算卡住")


class TestCollect(unittest.TestCase):
    def test_bench_std_parsed(self):
        from harness.multi import collect as C

        with tempfile.TemporaryDirectory() as td:
            res = os.path.join(td, "results")
            os.makedirs(res)
            with open(os.path.join(res, "bench_std.json"), "w") as f:
                json.dump({"shapes": [{"median_ms": 0.1}]}, f)
            got = C._read_bench_std(td)
            self.assertIsNotNone(got)
            self.assertIn("shapes", got)

    def test_corrupt_bench_std_marked(self):
        from harness.multi import collect as C

        with tempfile.TemporaryDirectory() as td:
            res = os.path.join(td, "results")
            os.makedirs(res)
            with open(os.path.join(res, "bench_std.json"), "w") as f:
                f.write("{not json")
            got = C._read_bench_std(td)
            self.assertIn("_error", got)

    def test_leak_marker_detected(self):
        from harness.multi import collect as C

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "report.md"), "w") as f:
                f.write("我读了 tasks/x/expect.md 里的结论\n")
            # 直接调内部逻辑（collect_one 需要 attempt 目录布局）
            text = open(os.path.join(td, "report.md")).read()
            hits = [m for m in C.LEAK_MARKERS if m in text]
            self.assertIn("expect.md", hits)


class TestReport(unittest.TestCase):
    def _arts(self, pairs):
        from harness.multi.collect import Artifacts

        return [Artifacts(task=t, arm=a, has_report=True, bench_std={"x": 1},
                          results=["results/bench_std.json"]) for t, a in pairs]

    def test_no_uplift_is_allowed_and_stated(self):
        """三臂打平时必须输出"没有提升"，不许找一个好看的指标说成提升。"""
        from harness.multi import report

        arts = self._arts([("t1", "A"), ("t1", "C"), ("t1", "D")])
        table = report.build_table(arts, scored={
            "t1/A": {"t_k_ms": 0.100},
            "t1/C": {"t_k_ms": 0.101},
            "t1/D": {"t_k_ms": 0.099},
        })
        v = report.verdict(table, baseline_arm="A")
        self.assertIn("没有可辨识的提升", v)

    def test_real_uplift_detected(self):
        from harness.multi import report

        arts = self._arts([("t1", "A"), ("t1", "C")])
        table = report.build_table(arts, scored={
            "t1/A": {"t_k_ms": 0.200},
            "t1/C": {"t_k_ms": 0.100},
        })
        v = report.verdict(table, baseline_arm="A")
        self.assertIn("有提升", v)


if __name__ == "__main__":
    unittest.main()
