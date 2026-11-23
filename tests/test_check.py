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
