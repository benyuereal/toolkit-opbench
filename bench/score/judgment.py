"""L3/L5 判定题打分。

不跑 bench，只看结论与证据：
  100  结论正确 + 有实测证据 + 说清机制
   60  结论正确但证据不足
   20  明确说"未验证/不知道"（诚实）
    0  结论错误或编造证据（并计入幻觉）
"""


def score(verdict: str, evidence: list[str], defn) -> tuple[int, str]:
    """返回 (分数, 理由)。defn 提供 expected_verdict / verdict_keywords /
    forbidden_claims / evidence_required。"""
    raise NotImplementedError


def is_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """命中 forbidden_claims 即视为编造（如 L5 题里说"可以融合"）。"""
    raise NotImplementedError
