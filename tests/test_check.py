"""bench/check 的单测（泄漏 / 幻觉 / 完整性）。

不依赖 GPU。幻觉检测里"import 真存在"这一条真的去 import，
这是刻意的：验证器自己必须先验证得住。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from bench.check import hallucination, integrity, leak

REGISTRY = "/workspace/toolkit_skills/data/ops/libraries-registry.yaml"


class TestLeak(unittest.TestCase):
    def test_scans_fingerprints(self):
        hits = leak.scan("这里直接写了 fuse_groupgemm_permute_quant")
        self.assertEqual(hits, ["fuse_groupgemm_permute_quant"])

    def test_empty_is_clean(self):
        self.assertEqual(leak.scan("我什么也没看到"), [])

    def test_multiple_hits(self):
        hits = leak.scan("49.8 TF 和 560 TF")
        self.assertEqual(set(hits), {"49.8 TF", "560 TF"})

    def test_case_sensitive(self):
        """指纹是函数名/字段名，大小写不同就是另一个东西。"""
        self.assertEqual(leak.scan("HARD_PRECONDITIONS"), [])

    def test_scan_dir_only_text(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "results"))
            with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
                f.write("命中 l1_exempt_reason")
            with open(os.path.join(d, "results", "x.bin"), "wb") as f:
                f.write(b"l1_exempt_reason")
            hits = leak.scan_dir(d)
            self.assertIn("report.md", hits)
            self.assertNotIn("results/x.bin", hits)


class TestIntegrity(unittest.TestCase):
    def test_hash_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "task.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write("abc")
            h1 = integrity.hash_file(p)
            with open(p, "w", encoding="utf-8") as f:
                f.write("abd")
            self.assertNotEqual(h1, integrity.hash_file(p))

    def test_verify_task_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "task.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write("题面")
            h = integrity.hash_file(p)
            self.assertTrue(integrity.verify_task_unchanged(d, h))
            with open(p, "w", encoding="utf-8") as f:
                f.write("题面改了")
            self.assertFalse(integrity.verify_task_unchanged(d, h))

    def test_missing_task_md_is_not_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(integrity.verify_task_unchanged(d, "deadbeef"))

    def test_snapshot_and_diff(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.py")
            with open(a, "w", encoding="utf-8") as f:
                f.write("x=1")
            before = integrity.hash_dir(d)
            with open(a, "w", encoding="utf-8") as f:
                f.write("x=2")
            with open(os.path.join(d, "b.py"), "w", encoding="utf-8") as f:
                f.write("y")
            os.unlink(a)
            after = integrity.hash_dir(d)
            problems = integrity.diff_hashes(before, after)
            self.assertTrue(any("删除" in p for p in problems))
            self.assertTrue(any("新文件" in p for p in problems))

    def test_function_snapshot(self):
        def fn1():
            pass

        ns = {"fn": fn1}
        snap = integrity.snapshot_functions(ns, ["fn", "missing"])
        self.assertIn("fn", snap)
        self.assertNotIn("missing", snap)
        self.assertEqual(integrity.check_functions_unchanged(snap, ns), [])

        def fn2():
            pass

        self.assertTrue(integrity.check_functions_unchanged(snap, {"fn": fn2}))
        self.assertTrue(integrity.check_functions_unchanged(snap, {}))

    def test_threads(self):
        self.assertTrue(integrity.check_no_background_threads(2, 2))
        self.assertTrue(integrity.check_no_background_threads(2, 1))
        self.assertFalse(integrity.check_no_background_threads(1, 3))

    def test_lazy_meta_tensor_detected(self):
        import torch

        t = torch.empty(4, 4, device="meta")
        problems = integrity.check_no_lazy_tensors({"out": t})
        self.assertTrue(any("meta" in p for p in problems))

    def test_real_tensor_ok(self):
        import torch

        problems = integrity.check_no_lazy_tensors({"out": torch.zeros(2, 2)})
        self.assertEqual(problems, [])

    def test_shape_claim(self):
        self.assertEqual(integrity.check_shape_claim({"M": [4096]}, {"M": [4096]}), [])
        probs = integrity.check_shape_claim({"M": [4096]}, {"M": [1024]})
        self.assertTrue(any("1024" in p for p in probs))
        probs = integrity.check_shape_claim({"M": [4096]}, {})
        self.assertTrue(any("没有这个 shape" in p for p in probs))


@unittest.skipUnless(os.path.isfile(REGISTRY), "registry 未挂载")
class TestHallucinationLibrary(unittest.TestCase):
    def test_real_library(self):
        ok, why = hallucination.verify_library_installed("aiter", REGISTRY)
        self.assertTrue(ok, why)

    def test_alias_recognized(self):
        ok, _ = hallucination.verify_library_installed("flash_attn", REGISTRY)
        self.assertTrue(ok)

    def test_ghost_library(self):
        ok, why = hallucination.verify_library_installed("ghostlib", REGISTRY)
        self.assertFalse(ok)
        self.assertIn("编造", why)

    def test_removed_library_has_reason(self):
        ok, why = hallucination.verify_library_installed("hipblaslt", REGISTRY)
        self.assertFalse(ok)
        self.assertIn("移除", why)
        self.assertIn("triton", why)

    def test_not_installed(self):
        ok, why = hallucination.verify_library_installed("in-house", REGISTRY)
        self.assertFalse(ok)


class TestHallucinationImports(unittest.TestCase):
    def test_existing_path(self):
        ok, why = hallucination.verify_import_path("json.dumps")
        self.assertTrue(ok, why)

    def test_missing_attr(self):
        ok, why = hallucination.verify_import_path("json.no_such_fn")
        self.assertFalse(ok)
        self.assertIn("没有", why)

    def test_missing_module(self):
        ok, _ = hallucination.verify_import_path("no_such_module_zzz.fn")
        self.assertFalse(ok)

    def test_bare_name_rejected(self):
        ok, why = hallucination.verify_import_path("dumps")
        self.assertFalse(ok)


class TestHallucinationNumbers(unittest.TestCase):
    def test_number_found(self):
        claims = {"numbers": [{"kind": "ratio", "value": 1.25, "source": "r"}]}
        self.assertEqual(
            hallucination.verify_numbers_against_results(claims, {"a": {"ratio": 1.25}}), []
        )

    def test_number_missing(self):
        claims = {"numbers": [{"kind": "ratio", "value": 1.25, "source": "r"}]}
        probs = hallucination.verify_numbers_against_results(claims, {"a": {"ratio": 0.9}})
        self.assertEqual(len(probs), 1)

    def test_rounding_tolerated(self):
        claims = {"numbers": [{"kind": "ms", "value": 1.2345, "source": "r"}]}
        self.assertEqual(
            hallucination.verify_numbers_against_results(claims, {"a": 1.23449}), []
        )

    def test_check_solution_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "results"))
            with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
                f.write("调用 aiter.asm_fmoe_a8(x)，ratio = 2.0x\n")
            with open(os.path.join(d, "results", "r.json"), "w", encoding="utf-8") as f:
                json.dump({"ratio": 2.0}, f)
            out = hallucination.check_solution(None, d, REGISTRY)
            self.assertGreater(out["checked"], 0)
            self.assertEqual(out["failed"], [], out["failed"])
            self.assertEqual(out["rate"], 0.0)


class TestRunHelpers(unittest.TestCase):
    """执行器的纯逻辑部分：预算解析、任务目录解析、题目 id 归一。"""

    def test_stated_budget_parsed(self):
        from harness import run

        self.assertEqual(run.stated_budget_s("## 约束\n\n- 时间预算：**30 分钟**\n"), 1800)
        self.assertEqual(run.stated_budget_s("时间预算: 10 分钟"), 600)
        self.assertIsNone(run.stated_budget_s("没有写预算"))

    def test_task_dir_by_dirname_and_by_id(self):
        from harness import run

        a = run.task_dir("l5-asm-fmoe-a8")     # 目录名
        b = run.task_dir("moe-asm-fmoe-l5")    # definition.id
        self.assertEqual(a, b)

    def test_task_dir_unknown(self):
        from harness import run

        with self.assertRaises(FileNotFoundError):
            run.task_dir("no-such-task-xyz")

    def test_budget_below_stated_is_refused(self):
        from harness import run

        with self.assertRaises(ValueError) as ctx:
            run.run_one("moe-asm-fmoe-l5", "B", 60)
        self.assertIn("低于题面", str(ctx.exception))

    def test_suite_expands_using_definition_id(self):
        """suite 里写目录名，队列里也必须归一成 definition.id。"""
        import tempfile
        from harness import run

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("name: t\ntasks:\n  - id: l5-asm-fmoe-a8\n    budget: 600\n")
            path = f.name
        try:
            q = run.prepare_suite(path, ["A"])
        finally:
            os.unlink(path)
        self.assertEqual([x["task"] for x in q], ["moe-asm-fmoe-l5"])


if __name__ == "__main__":
    unittest.main()


class TestArms(unittest.TestCase):
    """三臂的上下文注入：只差「一行资源声明」是实验有效性的前提。"""

    def test_prompts_share_task_text(self):
        from harness.arms import a, b, c

        task = "# 题目\n\n背景...\n"
        for mod in (a, b, c):
            self.assertIn(task.strip(), mod.build_prompt(task))

    def test_prompts_differ_by_one_declaration(self):
        from harness.arms import a, b, c

        task = "# 题目"
        pa, pb, pc = (m.build_prompt(task) for m in (a, b, c))
        # 去掉最后一段声明后必须完全一致
        self.assertEqual(pa.rsplit("---", 1)[0].strip(), task)
        self.assertEqual(pb.rsplit("---", 1)[0].strip(), task)
        self.assertEqual(pc.rsplit("---", 1)[0].strip(), task)
        self.assertNotEqual(pa, pb)
        self.assertNotEqual(pb, pc)

    def test_a_arm_does_not_leak_skill_words(self):
        from harness.arms import a

        p = a.build_prompt("# 题目")
        for word in ("toolkit_skills", "atlas", "scenario", "failure-mode"):
            self.assertNotIn(word, p)

    def test_a_arm_stages_nothing(self):
        import tempfile
        from harness.arms import a

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(a.stage_files(d), [])

    def test_b_arm_copies_registry_not_symlinks(self):
        import tempfile
        from harness.arms import b

        with tempfile.TemporaryDirectory() as d:
            staged = b.stage_files(d)
            self.assertEqual(staged, ["libraries-registry.yaml"])
            p = os.path.join(d, "libraries-registry.yaml")
            self.assertTrue(os.path.isfile(p))
            self.assertFalse(os.path.islink(p), "软链会让 B 臂顺着路径变成 C 臂")
            self.assertEqual(os.stat(p).st_mode & 0o777, 0o444)

    def test_b_prompt_mentions_registry_only(self):
        from harness.arms import b

        p = b.build_prompt("# 题目")
        self.assertIn("libraries-registry.yaml", p)
        self.assertNotIn("toolkit_skills", p)

    @unittest.skipUnless(os.path.isdir("/workspace/toolkit_skills"), "skill 未挂载")
    def test_c_arm_symlinks_skill(self):
        import tempfile
        from harness.arms import c

        with tempfile.TemporaryDirectory() as d:
            mounted = c.mount(d)
            self.assertEqual(mounted, ["skill"])
            link = os.path.join(d, "skill")
            self.assertTrue(os.path.islink(link))
            self.assertEqual(os.path.realpath(link), os.path.realpath(c.SKILL_ROOT))

    @unittest.skipUnless(os.path.isdir("/workspace/toolkit_skills"), "skill 未挂载")
    def test_c_skill_version_recorded(self):
        from harness.arms import c

        v = c.skill_version()
        self.assertEqual(v["root"], c.SKILL_ROOT)
        self.assertIsInstance(v["git_head"], str)

    def test_c_missing_skill_raises(self):
        import tempfile
        from harness.arms import c

        with tempfile.TemporaryDirectory() as d, \
             unittest.mock.patch.object(c, "SKILL_ROOT", "/nonexistent/skill"):
            with self.assertRaises(FileNotFoundError):
                c.mount(d)


class TestArmBudgetComparability(unittest.TestCase):
    """三臂只许差上下文，不许差预算。"""

    def test_mismatched_usd_refused(self):
        import json as _json
        import tempfile
        from harness import run

        runs = tempfile.mkdtemp()
        d = os.path.join(runs, "arm-A", "t9", "attempt-1")
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            _json.dump({"budget_s": 1800, "max_usd": 3.0}, f)
        with unittest.mock.patch.object(run, "RUNS_DIR", runs):
            with self.assertRaises(ValueError) as ctx:
                run.assert_arms_comparable("t9", 1800, 5.0)
            self.assertIn("max_usd", str(ctx.exception))
            # 一致时不报错
            self.assertEqual(run.assert_arms_comparable("t9", 1800, 3.0), [])

    def test_mismatched_wall_budget_refused(self):
        import json as _json
        import tempfile
        from harness import run

        runs = tempfile.mkdtemp()
        d = os.path.join(runs, "arm-B", "t9", "attempt-1")
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            _json.dump({"budget_s": 600, "max_usd": 3.0}, f)
        with unittest.mock.patch.object(run, "RUNS_DIR", runs):
            with self.assertRaises(ValueError):
                run.assert_arms_comparable("t9", 1800, 3.0)

    def test_no_previous_runs_is_fine(self):
        import tempfile
        from harness import run

        with unittest.mock.patch.object(run, "RUNS_DIR", tempfile.mkdtemp()):
            self.assertEqual(run.assert_arms_comparable("none", 1800, 3.0), [])
