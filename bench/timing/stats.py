"""统计工具：MAD、显著性、跨形状几何平均。"""
import math


def median(xs: list[float]) -> float:
    """中位数。空列表抛错——返回 0 会让"没有样本"伪装成"很快"。"""
    if not xs:
        raise ValueError("median() 需要至少一个样本")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def mad(xs: list[float]) -> float:
    """中位绝对偏差。比标准差抗离群。

    为什么要 MAD 而不是标准差：计时样本里偶发的调度抖动会制造一两个
    离群点，标准差被它拉大、均值被它拖走，于是"两次都量了 50 次"看起来
    却"差得不多"。MAD 只关心中位数附近的散布。
    """
    if not xs:
        raise ValueError("mad() 需要至少一个样本")
    m = median(xs)
    return median([abs(x - m) for x in xs])


def cv(xs: list[float]) -> float:
    """变异系数 (MAD / median)。用 MAD 而不是标准差做分子，同 mad() 的理由。

    median 为 0 时返回 inf（不是 0）：那是"量不出来"而不是"很稳"。
    """
    if not xs:
        raise ValueError("cv() 需要至少一个样本")
    m = median(xs)
    if m == 0:
        return float("inf")
    return mad(xs) / abs(m)


def significant(a: float, b: float, mad_a: float, mad_b: float, k: float = 3.0) -> bool:
    """|a-b| > k × max(MAD) 才算显著。别拿噪声当结论。

    两个 MAD 都为 0 时：只要 a != b 就显著（两批样本内部完全一致，
    差异不可能来自散布）。这一条要显式写，否则 max(0,0)=0 会让
    "a-b > 0" 退化成浮点比较。
    """
    scale = max(abs(mad_a), abs(mad_b))
    if scale == 0:
        return a != b
    return abs(a - b) > k * scale


def geomean(xs: list[float]) -> float:
    """跨形状几何平均。防"挑一个好看的形状报数"。

    要求全部为正：几何平均对 0 与负数无定义，而"跳过坏样本再平均"
    正是我们想禁止的。要么全报，要么整组作废。
    """
    if not xs:
        raise ValueError("geomean() 需要至少一个样本")
    for x in xs:
        if not math.isfinite(x) or x <= 0:
            raise ValueError(f"几何平均只接受正数，实际 {x!r}")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def overlap_ratio(t_serial: float, t_parallel: float,
                  t_a: float, t_b: float,
                  min_ratio: float = 0.2, max_ratio: float = 5.0) -> float:
    """重叠度 = (串行 - 并行) / min(t_a, t_b)。分母必须用 min——物理上最多只能
    隐藏较小那一侧的全部耗时。另需校验两侧量级比 ∈[0.2,5]，否则拒出结论。

    为什么分母必须是 min：
      两侧分别耗时 10 ms 与 2 ms。最理想的重叠只能藏掉 2 ms，
      所以上限 = 2ms/2ms = 100%。若用 max（10ms）做分母，
      同样的数据只能算出 20%——看起来"重叠很差"，实际已经打满。
      当年用错分母，两次结论直接相反。

    为什么量级比要校验：10 ms 与 0.1 ms 的两段，就算完全重叠，
    总耗时也只是从 10.1 降到 10.0，测出来的"重叠度"完全被噪声支配。
    这种数据算出来的任何比例都是噪声，必须拒。

    返回 0–1（可略小于 0：并行比串行还慢，说明负重叠）。
    """
    for name, v in (("t_serial", t_serial), ("t_parallel", t_parallel),
                    ("t_a", t_a), ("t_b", t_b)):
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} 必须是非负有限数，实际 {v!r}")
    if t_a <= 0 or t_b <= 0:
        raise ValueError("两侧耗时都必须为正——0 意味着那一段没测到")
    if t_serial <= 0:
        raise ValueError("串行总耗时必须为正")

    lo, hi = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
    if hi / lo > max_ratio or hi / lo < min_ratio:
        raise ValueError(
            f"两侧量级比 {hi / lo:.2f} 超出 [{min_ratio}, {max_ratio}]："
            f"{t_a} vs {t_b}。这种数据算出的重叠度被噪声支配，拒绝出结论"
        )

    return (t_serial - t_parallel) / min(t_a, t_b)
