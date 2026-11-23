"""校准度：它宣称的达成率与实测差多少。

这是 49.8 那次事故的直接产物——宣称"到顶"而实际有 11 倍空间。
判据：|宣称 - 实测| <= 10pt。

判定口径：
  * 宣称与实测都按「占峰值百分比」计（0-100）。
  * 抽不到宣称时返回 None，由上层按「不可校准」处理，不要当成 0 误差。
"""

from __future__ import annotations

import re

# 带这些词的句子才可能是在谈达成率；否则文中的百分数可能是别的东西（占比、误差）。
_ACH_KEYWORDS = (
    "达成",
    "到顶",
    "达到了",
    "利用率",
    "峰值",
    "utilization",
    "of peak",
    "achieved",
    "roofline",
)

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_SENT_SPLIT = re.compile(r"[。\n；;.!?]")


def achievement_pct(measured: float, peak: float) -> float | None:
    """实测占峰值的百分比。peak 未知或 <=0 时返回 None（不猜）。"""
    try:
        m = float(measured)
        p = float(peak)
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    return m / p * 100.0


def claimed_achievement(text: str, peak: float, measured: float) -> float | None:
    """从报告文本里抽"达成率"声明（如"已到 92%"）。抽不到返回 None。

    文本里出现多个百分数时，取与实测达成率最接近的那个（peak <= 0 时取第一个）。
    这么做的理由：一份报告常同时提到"占比 30%""误差 1.2%"，直接取第一个会抽错。
    """
    if not text:
        return None

    candidates: list[float] = []
    for sent in _SENT_SPLIT.split(text):
        if not sent:
            continue
        if any(k in sent.lower() for k in _ACH_KEYWORDS):
            candidates.extend(float(m.group(1)) for m in _PCT.finditer(sent))

    if not candidates:
        candidates = [float(m.group(1)) for m in _PCT.finditer(text)]
    if not candidates:
        return None

    ref = achievement_pct(measured, peak)
    if ref is None or len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda c: abs(c - ref))


def calibration_error(claimed: float, measured: float, peak: float | None = None) -> float:
    """宣称与实测的差（百分点）。

    peak 为 None 时，claimed 与 measured 都视为已经是百分比。
    """
    c = float(claimed)
    if peak is None:
        m = float(measured)
    else:
        m = achievement_pct(measured, peak)
        if m is None:
            raise ValueError(f"peak={peak} 无法计算达成率，不能做校准判定")
    return abs(c - m)


def is_calibrated(
    claimed: float,
    measured: float,
    tol_pt: float = 10.0,
    peak: float | None = None,
) -> bool:
    """|宣称 - 实测| <= tol_pt 且「宣称更乐观」的幅度也在容差内。

    单向也判：如果宣称比实测低得多（把有空间说成没空间），同样是失准——
    49.8 那次就是把 11 倍空间说成了到顶，方向相反但错误一样严重。
    """
    return calibration_error(claimed, measured, peak) <= float(tol_pt)
