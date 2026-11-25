"""env 404 回归 + L5 judgment 题可加载。

tosc.env 的 ANTHROPIC_BASE_URL 结尾带 `/`，原样传会让 Claude Code 拼出
//v1/messages → 404。这里固化"rstrip 必须生效"这条，防止回退。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from bench.data import definition as D
from bench.score import judgment


class TestEnvRStrip(unittest.TestCase):
    """ANTHROPIC_BASE_URL 尾斜杠必须被去掉，否则 404。"""

    def _load_from(self, content: str):
        import harness.run as run
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            with mock.patch.object(run, "ENV_FILE", path):
                return run._load_env()
        finally:
            os.unlink(path)

    def test_trailing_slash_stripped(self):
        env = self._load_from("ANTHROPIC_BASE_URL=http://10.20.100.15:4001/\n")
        self.assertFalse(env["ANTHROPIC_BASE_URL"].endswith("/"))
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://10.20.100.15:4001")

    def test_multiple_slashes_stripped(self):
        env = self._load_from("ANTHROPIC_BASE_URL=http://h:1//\n")
        self.assertFalse(env["ANTHROPIC_BASE_URL"].endswith("/"))

    def test_non_url_key_untouched(self):
        env = self._load_from("ANTHROPIC_AUTH_TOKEN=abc/\n")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "abc/")

    def test_export_prefix_and_quotes(self):
        env = self._load_from('export ANTHROPIC_BASE_URL="http://h:1/"\n')
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://h:1")


class TestL5JudgmentLoads(unittest.TestCase):
    """l5 题的 definition.json 必须能被 load + judgment 评分器消费。"""

    def _l5_defn(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(root, "tasks", "l5-asm-fmoe-a8", "definition.json")
        self.assertTrue(os.path.isfile(p), f"缺 {p}")
        return D.load(p)

    def test_l5_definition_loads(self):
        d = self._l5_defn()
        self.assertEqual(d.kind, "judgment")
        self.assertEqual(d.tier, "L5")
        self.assertTrue(d.verdict_keywords)
        self.assertTrue(d.forbidden_claims)
        self.assertTrue(d.evidence_required)

    def test_judgment_scores_consistent_with_defn(self):
        """用该题自己的 verdict_keywords 造合格回答，避免与答案口径脱节。

        这道题的 expected_verdict 已从 unavailable 改成 conditional
        （实测它有条件可用：显式给 solidx + 正确传 sorted_weights）。
        测试跟着 defn 走，硬编码"不可用"会随答案变化而失效。
        """
        d = self._l5_defn()
        kw = d.verdict_keywords[0]
        s_ok, why_ok = judgment.score(
            f"{kw}。实测需显式指定 mode 与 solidx 才能命中 config",
            ["error_output: RuntimeError: unsupport current input Mode: 0",
             "command: python3 probe.py --solidx 10000",
             "import aiter 调用 asm_fmoe_a8 后调用成功",
             "results/probe.json rel_l2=1.7e-3"],
            d,
        )
        self.assertGreaterEqual(s_ok, 60, why_ok)
        # 编造（命中 forbidden）
        s_bad, _ = judgment.score("可以一步融合", ["x"], d)
        self.assertEqual(s_bad, 0)


if __name__ == "__main__":
    unittest.main()
