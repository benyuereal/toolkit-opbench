"""统计工具：MAD、显著性、跨形状几何平均。"""
import math


def median(xs: list[float]) -> float:
    raise NotImplementedError


def mad(xs: list[float]) -> float:
    """中位绝对偏差。比标准差抗离群。"""
    raise NotImplementedError


def significant(a: float, b: float, mad_a: float, mad_b: float, k: float = 3.0) -> bool:
    """|a-b| > k × max(MAD) 才算显著。别拿噪声当结论。"""
    raise NotImplementedError


def geomean(xs: list[float]) -> float:
    """跨形状几何平均。防"挑一个好看的形状报数"。"""
    raise NotImplementedError


def overlap_ratio(t_serial: float, t_parallel: float, t_a: float, t_b: float) -> float:
    """重叠度 =(串行-并行)/min(t_a,t_b)。分母必须用 min——物理上最多只能
    隐藏较小那一侧的全部耗时。另需校验两侧量级比 ∈[0.2,5]，否则拒出结论。"""
    raise NotImplementedError
