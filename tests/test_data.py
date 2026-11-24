"""bench/data 的单测。不依赖 GPU、不依赖 pytest。

    python3 -m unittest discover -s tests
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from bench.data import definition as defn_mod
from bench.data import solution as sol_mod
from bench.data import trace as trace_mod
from bench.data import workload as wl_mod


def _kernel(**over):
    d = {
        "id": "gemm-bf16-l1",
        "kind": "kernel",
        "family": "gemm",
        "tier": "L1",
        "precision": ["bfloat16"],
        "axes": {"M": {"type": "var"}, "N": {"type": "const", "value": 4096}},
        "inputs": {
            "A": {"shape": ["M", "N"], "dtype": "bfloat16"},
            "B": {"shape": ["M", "N"], "dtype": "bfloat16"},
        },
        "outputs": {"C": {"shape": ["M", "N"], "dtype": "bfloat16"}},
        "tolerance": {"rtol": 0.02, "atol": 0.01},
        "reference": "reference.py",
        "dcu": {"peak_constant": "peak_achievable:bf16", "min_l2_flush_mb": 16},
    }
    d.update(over)
    return d


class TestDefinition(unittest.TestCase):
    def test_valid_kernel(self):
        self.assertEqual(defn_mod.validate(_kernel()), [])

    def test_const_var_axes(self):
        d = defn_mod.Definition(
            id="x", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var"}, "N": {"type": "const", "value": 8}},
            inputs={}, outputs={},
        )
        self.assertEqual(d.const_axes(), {"N": 8})
        self.assertEqual(d.var_axes(), ["M"])

    def test_kernel_needs_tolerance_and_reference(self):
        probs = defn_mod.validate(_kernel(tolerance=None, reference=None))
        self.assertTrue(any("tolerance" in p for p in probs))
        self.assertTrue(any("reference" in p for p in probs))

    def test_bad_kind_and_tier(self):
        probs = defn_mod.validate(_kernel(kind="op", tier="L9"))
        self.assertTrue(any("kind" in p for p in probs))
        self.assertTrue(any("tier" in p for p in probs))

    def test_unknown_field_caught(self):
        probs = defn_mod.validate(_kernel(tolerence={"rtol": 1}))
        self.assertTrue(any("未知字段" in p for p in probs))

    def test_axis_must_be_const_or_var(self):
        d = _kernel(axes={"M": {"type": "var"}, "N": {"type": "fixed", "value": 4}})
        self.assertTrue(any("type" in p for p in defn_mod.validate(d)))

    def test_const_axis_needs_positive_int(self):
        d = _kernel(axes={"M": {"type": "var"}, "N": {"type": "const", "value": -3}})
        self.assertTrue(any("正整数" in p for p in defn_mod.validate(d)))

    def test_shape_refs_declared_axis(self):
        d = _kernel(inputs={"A": {"shape": ["M", "Q"], "dtype": "bfloat16"}})
        self.assertTrue(any("未声明的轴" in p for p in defn_mod.validate(d)))

    def test_dtype_whitelist(self):
        d = _kernel(inputs={"A": {"shape": ["M", "N"], "dtype": "fp8"}})
        self.assertTrue(any("dtype" in p for p in defn_mod.validate(d)))

    def test_peak_constant_rejects_literal_in_id(self):
        d = _kernel(dcu={"peak_constant": "fp8_peak_nominal_560"})
        self.assertTrue(any("峰值写进了 id" in p for p in defn_mod.validate(d)))

    def test_peak_constant_must_exist_upstream(self):
        d = _kernel(dcu={"peak_constant": "peak_achievable:fp16"})
        probs = defn_mod.validate(d)
        self.assertTrue(any("常数" in p or "没有这个" in p for p in probs))

    def test_peak_constant_cannot_mix_caliber(self):
        # 49.8 是 impl_measured；按 achievable 引用必须被拒（当年就是这里错的）
        d = _kernel(dcu={"peak_constant": "impl_measured:fp8"})
        self.assertEqual(defn_mod.validate(d), [])
        wrong = defn_mod.validate(_kernel(dcu={"peak_constant": "peak_achievable:fp8"}))
        self.assertTrue(any("不存在" in p or "没有这个" in p for p in wrong))

    def test_judgment_definition(self):
        d = {
            "id": "moe-asm-fmoe-l5",
            "kind": "judgment",
            "family": "moe",
            "tier": "L5",
            "question": "asm 能不能用",
            "expected_verdict": "unavailable",
            "verdict_keywords": ["unsupport"],
            "forbidden_claims": ["可以融合"],
            "evidence_required": ["error_output", "command"],
        }
        self.assertEqual(defn_mod.validate(d), [])

    def test_judgment_requires_evidence(self):
        d = {
            "id": "x", "kind": "judgment", "family": "moe", "tier": "L5",
            "question": "q", "expected_verdict": "unavailable",
            "verdict_keywords": ["a"], "forbidden_claims": ["b"],
        }
        self.assertTrue(any("evidence_required" in p for p in defn_mod.validate(d)))

    def test_judgment_rejects_kernel_fields(self):
        d = {
            "id": "x", "kind": "judgment", "family": "moe", "tier": "L5",
            "question": "q", "expected_verdict": "unavailable",
            "verdict_keywords": ["a"], "forbidden_claims": ["b"],
            "evidence_required": ["c"], "tolerance": {"rtol": 1, "atol": 0},
        }
        self.assertTrue(any("未知字段" in p for p in defn_mod.validate(d)))

    def test_repo_templates_are_valid(self):
        """仓库里的真实题面必须过校验：模板坏了，照抄出来的 39 道题全坏。"""
        import glob
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for path in sorted(glob.glob(os.path.join(root, "tasks", "*", "definition.json"))):
            with self.subTest(path=path):
                d = defn_mod.load(path)
                wl = os.path.join(os.path.dirname(path), "workload.jsonl")
                if d.kind == "kernel" and os.path.isfile(wl):
                    ws = wl_mod.load(wl, d)
                    wl_mod.split(ws)

    def test_load_collects_all_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "definition.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(_kernel(kind="nope", tier="L9"), f)
            with self.assertRaises(ValueError) as ctx:
                defn_mod.load(p)
            self.assertIn("kind", str(ctx.exception))
            self.assertIn("tier", str(ctx.exception))


class TestWorkload(unittest.TestCase):
    def setUp(self):
        self.defn = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var", "domain": {"min": 1, "max": 10}},
                  "N": {"type": "const", "value": 4}},
            inputs={"A": {"shape": ["M", "N"], "dtype": "float32"}},
            outputs={}, tolerance={"rtol": 1e-2, "atol": 1e-2},
            reference="reference.py",
        )

    def _write(self, tmp, lines):
        p = os.path.join(tmp, "workload.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj) + "\n")
        return p

    def test_load_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                {"axes": {"M": 4}, "uuid": "pub", "split": "public"},
                {"axes": {"M": 7}, "uuid": "hold", "split": "holdout"},
            ])
            ws = wl_mod.load(p, self.defn)
            self.assertEqual(len(ws), 2)
            pub, hold = wl_mod.split(ws)
            self.assertEqual([w.uuid for w in pub], ["pub"])
            self.assertEqual([w.uuid for w in hold], ["hold"])

    def test_split_requires_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"axes": {"M": 4}, "uuid": "pub"}])
            ws = wl_mod.load(p)
            with self.assertRaises(ValueError):
                wl_mod.split(ws)

    def test_rejects_value_for_const_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                {"axes": {"M": 4, "N": 8}, "uuid": "a", "split": "public"},
            ])
            with self.assertRaises(ValueError):
                wl_mod.load(p, self.defn)

    def test_rejects_missing_var_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"axes": {}, "uuid": "a", "split": "public"}])
            with self.assertRaises(ValueError):
                wl_mod.load(p, self.defn)

    def test_rejects_bad_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                {"axes": {"M": 4}, "uuid": "a", "split": "private"},
            ])
            with self.assertRaises(ValueError):
                wl_mod.load(p, self.defn)

    def test_unknown_input_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                {"axes": {"M": 4}, "uuid": "a", "split": "public",
                 "inputs": {"A": {"type": "noise"}}},
            ])
            with self.assertRaises(ValueError):
                wl_mod.load(p, self.defn)

    def test_seed_is_stable(self):
        self.assertEqual(wl_mod._seed_of("abc"), wl_mod._seed_of("abc"))
        self.assertNotEqual(wl_mod._seed_of("abc"), wl_mod._seed_of("abd"))

    def test_generate_holdout_with_values_domain(self):
        d = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "var", "domain": {"values": [1, 2, 3, 4]}}},
            inputs={"A": {"shape": ["M"], "dtype": "float32"}},
            outputs={},
        )
        hs = wl_mod.generate_holdout(d, n=3, seed=7)
        self.assertEqual(len(hs), 3)
        self.assertTrue(all(h.is_holdout() for h in hs))
        self.assertEqual(len({tuple(h.axes.items()) for h in hs}), 3)

    def test_generate_holdout_without_var(self):
        d = defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "const", "value": 2}}, inputs={}, outputs={},
        )
        with self.assertRaises(ValueError):
            wl_mod.generate_holdout(d)


class TestSolution(unittest.TestCase):
    def test_kernel_requires_entrypoint(self):
        probs = sol_mod.validate({"task_id": "t", "kind": "kernel"})
        self.assertTrue(any("entrypoint" in p for p in probs))

    def test_valid(self):
        self.assertEqual(
            sol_mod.validate(
                {"task_id": "t", "kind": "kernel", "entrypoint": "solution.py:run"}
            ),
            [],
        )

    def test_artifacts_kernel_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = sol_mod.validate_artifacts(tmp, "kernel")
            self.assertTrue(any("solution.json" in p for p in problems))
            self.assertTrue(any("results" in p for p in problems))

    def test_artifacts_kernel_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "solution.json"), "w", encoding="utf-8") as f:
                json.dump({"task_id": "t", "kind": "kernel",
                           "entrypoint": "solution.py:run"}, f)
            with open(os.path.join(tmp, "results", "r.json"), "w", encoding="utf-8") as f:
                json.dump({"timing": {"median_ms": 1.0}, "shape": {"M": 4}}, f)
            self.assertEqual(sol_mod.validate_artifacts(tmp, "kernel"), [])

    def test_artifacts_kernel_result_without_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "results", "r.json"), "w", encoding="utf-8") as f:
                json.dump({"shape": {"M": 4}}, f)
            problems = sol_mod.validate_artifacts(tmp, "kernel")
            self.assertTrue(any("timing" in p for p in problems))

    def test_artifacts_judgment_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "scripts"))
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "report.md"), "w", encoding="utf-8") as f:
                f.write("# 结论\n\n不可用。\n\n## 证据\n\n报错原文见 results/err.log\n")
            self.assertEqual(sol_mod.validate_artifacts(tmp, "judgment"), [])

    def test_artifacts_judgment_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "scripts"))
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "report.md"), "w", encoding="utf-8") as f:
                f.write("不可用。\n")
            problems = sol_mod.validate_artifacts(tmp, "judgment")
            self.assertTrue(any("证据" in p for p in problems))

    def test_extract_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "scripts"))
            with open(os.path.join(tmp, "report.md"), "w", encoding="utf-8") as f:
                f.write(
                    "结论：不可用。\n\n调用 aiter.asm_fmoe_stage1(x) 报错。\n"
                    "库 lightop 可用。ratio = 1.25x\n"
                )
            with open(os.path.join(tmp, "scripts", "probe.py"), "w", encoding="utf-8") as f:
                f.write("import aiter\nfrom torch import nn\n")
            claims = sol_mod.extract_claims(tmp)
            self.assertIn("aiter.asm_fmoe_stage1", claims["import_paths"])
            self.assertIn("aiter", claims["libraries"])
            self.assertIn("lightop", claims["libraries"])
            self.assertTrue(any(n["kind"] == "ratio" and n["value"] == 1.25
                                for n in claims["numbers"]))


class TestTrace(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "trace.jsonl")
            t = trace_mod.Trace(
                task="t1", arm="A", started_at="2026-11-23T10:00:00",
                wall_time_s=12.5, exit_code=0, sclk_mhz=1200,
                libs={"torch": "2.11.0"}, artifacts=["report.md"],
            )
            trace_mod.append(p, t)
            trace_mod.append(p, t)
            got = trace_mod.load(p)
            self.assertEqual(len(got), 2)
            self.assertEqual(got[0].sclk_mhz, 1200)
            self.assertEqual(got[1].libs["torch"], "2.11.0")

    def test_requires_core_fields(self):
        with self.assertRaises(ValueError):
            trace_mod.from_dict({"task": "t"})

    def test_ignores_unknown_fields(self):
        t = trace_mod.from_dict({
            "task": "t", "arm": "A", "started_at": "x",
            "wall_time_s": 1.0, "exit_code": 0, "future_field": 1,
        })
        self.assertEqual(t.task, "t")

    def test_bad_line_reports_lineno(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not json}\n")
            with self.assertRaises(ValueError) as ctx:
                trace_mod.load(p)
            self.assertIn(":1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestCorrectionLog(unittest.TestCase):
    """修正记录必须入库且格式可解析——它是审计链的一环。"""

    def test_corrections_log_tracked_and_parsable(self):
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "runs", "corrections.log")
        self.assertTrue(os.path.isfile(path), "runs/corrections.log 必须存在")
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "--error-unmatch", "runs/corrections.log"],
            capture_output=True, text=True,
        )
        self.assertEqual(out.returncode, 0, "corrections.log 必须被 git 跟踪（不是运行产物）")
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        # 只有以 ISO 时间开头的行才是条目；其余是说明文字
        import re

        entries = [l for l in lines if re.match(r"^\d{4}-\d{2}-\d{2}T", l.strip())]
        self.assertTrue(entries, "至少一条修正记录")
        for line in entries:
            parts = line.split("|")
            self.assertGreaterEqual(len(parts), 5, line)
            self.assertTrue(parts[1].strip(), "条目必须写题目 id")


class TestIndexInputs(unittest.TestCase):
    """整数索引输入的范围与哨兵值必须由题面声明。

    实测事故：mla-sparse-l1 的 indices 默认生成 randint(0,8)，
    SKV=8192 却只出 0–7，且一个 -1 都没有——哨兵语义（本题三大考点之一）
    根本没被触发。
    """

    def _defn(self, **ind_spec):
        spec = {"shape": ["M", "TOPK"], "dtype": "int32"}
        spec.update(ind_spec)
        return defn_mod.Definition(
            id="t", kind="kernel", family="g", tier="L1",
            axes={"M": {"type": "const", "value": 16},
                  "SKV": {"type": "const", "value": 8192},
                  "TOPK": {"type": "const", "value": 256}},
            inputs={"idx": spec}, outputs={},
            tolerance={"rtol": 1e-2, "atol": 1e-2}, reference="reference.py",
        )

    def _build(self, d):
        w = wl_mod.Workload("u1", {}, {"idx": {"type": "random"}}, "public")
        return wl_mod.build_inputs(w, d)["idx"]

    def test_high_accepts_axis_name(self):
        d = self._defn(low=0, high="SKV")
        idx = self._build(d)
        self.assertGreaterEqual(int(idx.max()), 1000)   # 远超默认的 8
        self.assertLess(int(idx.max()), 8192)

    def test_invalid_ratio_inserts_sentinel(self):
        d = self._defn(low=0, high="SKV", invalid_ratio=0.3, invalid_value=-1)
        idx = self._build(d)
        frac = float((idx == -1).float().mean())
        self.assertGreater(frac, 0.2)
        self.assertLess(frac, 0.4)

    def test_no_sentinel_when_not_declared(self):
        d = self._defn(low=0, high="SKV")
        idx = self._build(d)
        self.assertEqual(int((idx < 0).sum()), 0)

    def test_bad_axis_raises(self):
        d = self._defn(low=0, high="NOSUCH")
        with self.assertRaises(ValueError):
            self._build(d)

    def test_high_must_exceed_low(self):
        d = self._defn(low=100, high=10)
        with self.assertRaises(ValueError):
            self._build(d)

    def test_reproducible_with_sentinel(self):
        import torch
        d = self._defn(low=0, high="SKV", invalid_ratio=0.3)
        self.assertTrue(torch.equal(self._build(d), self._build(d)))
