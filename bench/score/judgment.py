"""L3/L5 判定题打分。

不跑 bench，只看结论与证据：
  100  结论正确 + 有实测证据 + 说清机制
   60  结论正确但证据不足
   20  明确说"未验证/不知道"（诚实）
    0  结论错误或编造证据（并计入幻觉）

为什么有 20 分这一档：本项目最大的风险不是不够快，而是**编一个看起来合理的结论**。
历史上出现过三次（in-house 29 条、两个不存在的库、把实现达成值当硬件峰值）。
把"我不知道"给到 20 分，是为了让诚实比臆断得分高——但永远低于有证据的正确答案。
"""

from __future__ import annotations

# 表达"没测/不确定"的标记。命中即视为诚实声明，给 20 分而不是 0 分。
HONEST_MARKERS = (
    "未测定",
    "未验证",
    "未做验证",
    "不确定",
    "不知道",
    "暂无证据",
    "没测",
    "无法确认",
)

SCORE_FULL = 100
SCORE_THIN = 60
SCORE_HONEST = 20
SCORE_ZERO = 0


def _get(defn, key: str, default=None):
    """defn 可能是 dict，也可能是 dataclass/对象。两种都支持。"""
    if defn is None:
        return default
    if isinstance(defn, dict):
        return defn.get(key, default)
    return getattr(defn, key, default)


def is_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """命中 forbidden_claims 即视为编造（如 L5 题里说"可以融合"）。

    返回命中的片段列表；空列表表示没命中。ASCII 部分不区分大小写。
    """
    if not text or not forbidden:
        return []
    lowered = text.lower()
    hits = []
    for phrase in forbidden:
        if not phrase:
            continue
        p = str(phrase)
        if p.lower() in lowered:
            hits.append(p)
    return hits


def score(verdict: str, evidence: list[str], defn) -> tuple[int, str]:
    """返回 (分数, 理由)。defn 提供 expected_verdict / verdict_keywords /
    forbidden_claims / evidence_required。
    """
    text = (verdict or "").strip()
    keywords = _get(defn, "verdict_keywords", []) or []
    forbidden = _get(defn, "forbidden_claims", []) or []
    need = int(_get(defn, "evidence_required", 1) or 1)
    evidence = [e for e in (evidence or []) if str(e).strip()]

    if not text:
        return SCORE_ZERO, "结论为空"

    hits = is_forbidden(text, forbidden)
    if hits:
        return SCORE_ZERO, f"命中被禁止的主张 {hits[:3]}，视为编造证据"

    matched = [k for k in keywords if k and str(k) in text]
    honest = [m for m in HONEST_MARKERS if m in text]

    if not matched and not honest:
        return SCORE_ZERO, "结论既未命中判据关键词，也未声明不确定（疑似臆断）"

    if not matched and honest:
        return SCORE_HONEST, f"明确声明「{honest[0]}」，诚实但未给出结论"

    if len(evidence) < need:
        return (
            SCORE_THIN,
            f"结论方向正确（命中 {matched[:2]}），但证据 {len(evidence)} 条少于要求的 {need} 条",
        )

    return (
        SCORE_FULL,
        f"结论正确（命中 {matched[:2]}）且给出 {len(evidence)} 条证据",
    )
