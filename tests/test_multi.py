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


class TestPreflightGpuSufficiency(unittest.TestCase):
    """显存/占用判据：够用就放行，不够才拦。

    ## 事故（2026-11-26 09:41）

    卡 0/3 被占到 67%，卡 4–7 各空 139 GB。三题并行只要 3 张，完全跑得
    起来，但旧判据"任一卡忙就拦"把整轮挡下了。本机 8 张里长期只有 4 张
    可用，这条判据等于永远跑不了——比不检查还糟，因为它看起来像在工作。
    """

    def _patch(self, pcts, vram, levels=None):
        from unittest import mock

        from bench.hw import dcu

        lv = levels or [(8, 1300)] * len(pcts)
        return (
            mock.patch.object(dcu, "_busy_pcts", return_value=pcts),
            mock.patch.object(dcu, "_free_vram_gb", return_value=vram),
            mock.patch.object(dcu, "_sclk_levels", return_value=lv),
        )

    def test_enough_free_cards_passes_despite_busy_ones(self):
        from harness.multi import preflight

        p1, p2, p3 = self._patch(
            [67.0, 5.0, 14.0, 67.0, 0.0, 0.0, 0.0, 0.0],
            {0: 5.0, 1: 6.9, 2: 6.9, 3: 6.4,
             4: 139.0, 5: 139.0, 6: 139.0, 7: 139.0},
        )
        with p1, p2, p3:
            problems, warns, free = preflight.check_gpu(need=3)
        self.assertEqual(problems, [], f"有 4 张可用却报错：{problems}")
        self.assertEqual(free, [4, 5, 6, 7])
        self.assertTrue(any("够用" in w for w in warns), warns)

    def test_not_enough_free_cards_blocked(self):
        from harness.multi import preflight

        p1, p2, p3 = self._patch(
            [0.0, 67.0, 67.0, 67.0, 0.0, 0.0, 0.0, 0.0],
            {0: 139.0, 1: 5.0, 2: 5.0, 3: 5.0,
             4: 5.0, 5: 5.0, 6: 5.0, 7: 5.0},
        )
        with p1, p2, p3:
            problems, _w, free = preflight.check_gpu(need=3)
        self.assertTrue(problems, "只有 1 张可用却要 3 张，应当拦下")
        self.assertEqual(free, [0])

    def test_serial_needs_only_one(self):
        from harness.multi import preflight

        p1, p2, p3 = self._patch(
            [0.0] * 4 + [99.0] * 4,
            {0: 139.0, 1: 139.0, 2: 139.0, 3: 139.0,
             4: 139.0, 5: 139.0, 6: 139.0, 7: 139.0},
        )
        with p1, p2, p3:
            problems, _w, free = preflight.check_gpu(need=1)
        self.assertEqual(problems, [])
        self.assertEqual(len(free), 4)

    def test_clock_warning_only_among_usable_cards(self):
        """卡 0 是档位 8、卡 4 是档位 1 是常态，不能因此永久报警。"""
        from harness.multi import preflight

        levels = [(8, 1300), (8, 1300), (8, 1300), (8, 1300),
                  (1, 600), (1, 600), (1, 600), (1, 600)]
        p1, p2, p3 = self._patch(
            [99.0] * 4 + [0.0] * 4,
            {i: (5.0 if i < 4 else 139.0) for i in range(8)},
            levels=levels,
        )
        with p1, p2, p3:
            _p, warns, free = preflight.check_gpu(need=3)
        self.assertEqual(free, [4, 5, 6, 7])
        self.assertFalse(
            [w for w in warns if "档位不一致" in w],
            f"可用的 4 张档位一致（都是 1），不该报警：{warns}",
        )

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


class TestResumeLogic(unittest.TestCase):
    """claude -p 是单轮语义：说完不带 tool_use 的话就退出。

    实测 D 臂 281s/3600s、C 臂 462s/3600s 就这样"成功"了，交付物为空。
    这些测试锁住续跑机制，防止再退化成"说了就等于做了"。
    """

    def test_has_deliverable_requires_real_file(self):
        from harness.run import _has_deliverable

        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(_has_deliverable(td), "空目录不该算有交付物")
            os.makedirs(os.path.join(td, "results"))
            self.assertFalse(_has_deliverable(td), "空 results/ 不算")
            with open(os.path.join(td, "results", "bench_std.json"), "w") as f:
                f.write("{}")
            self.assertTrue(_has_deliverable(td))

    def test_report_md_alone_counts(self):
        from harness.run import _has_deliverable

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "report.md"), "w") as f:
                f.write("# 结论\n")
            self.assertTrue(_has_deliverable(td))

    def test_deliverable_sig_changes_on_write(self):
        """续跑靠指纹判断"这轮有没有新东西"，指纹必须对写入敏感。"""
        import time as _t

        from harness.run import _deliverable_sig

        with tempfile.TemporaryDirectory() as td:
            s0 = _deliverable_sig(td)
            _t.sleep(0.01)
            with open(os.path.join(td, "notes.md"), "w") as f:
                f.write("x")
            s1 = _deliverable_sig(td)
            self.assertNotEqual(s0, s1, "写了文件指纹没变，会误判成卡住")

    def test_build_cmd_passes_resume(self):
        from harness.run import build_cmd

        cmd = build_cmd("/tmp/x", 100, None, resume="abc-123")
        self.assertIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("--resume") + 1], "abc-123")

    def test_build_cmd_no_resume_by_default(self):
        from harness.run import build_cmd

        self.assertNotIn("--resume", build_cmd("/tmp/x", 100, None))

    def test_continue_prompt_demands_artifacts(self):
        """续跑提示必须点名交付物，否则模型会继续说下一句计划。"""
        from harness.run import CONTINUE_PROMPT

        self.assertIn("bench_std.json", CONTINUE_PROMPT)
        self.assertIn("report.md", CONTINUE_PROMPT)

    def test_resume_limits_sane(self):
        from harness.run import MAX_IDLE_RESUMES, MAX_RESUMES

        self.assertGreater(MAX_RESUMES, 0)
        self.assertLess(MAX_IDLE_RESUMES, MAX_RESUMES)
