"""校准度：它宣称的达成率与实测差多少。

这是 49.8 那次事故的直接产物——宣称"到顶"而实际有 11 倍空间。
判据：|宣称 - 实测| <= 10pt。
"""


def claimed_achievement(text: str, peak: float, measured: float) -> float | None:
    """从报告文本里抽"达成率"声明（如"已到 92%"）。抽不到返回 None。"""
    raise NotImplementedError


def calibration_error(claimed: float, measured: float) -> float:
    raise NotImplementedError


def is_calibrated(claimed: float, measured: float, tol_pt: float = 10.0) -> bool:
    raise NotImplementedError
