"""可读分：0.5 = 与基线持平，1 = 光速。

公式参考 SOL-ExecBench 的 sol_score，但**只用于报告，不用于奖励**——
它在接近光速处会饱和，用它比高下会掩盖差异。比较三臂请用 bench/score/ratio.py。
"""

from __future__ import annotations

import math

_EPS = 1e-9


def score(t_k_ms: float, t_b_ms: float, t_sol_ms: float) -> float:
    """S = 1 / (1 + (T_k - T_sol) / (T_b - T_sol))。

    含义：0.5 = 与基线持平，1.0 = 达到光速。

    两种退化情况按物理意义处理，不抛异常（这两条是报告里必须单独说明的）：
      * T_b == T_sol：基线已经等于光速，公式分母为 0。
        此时 T_k <= T_b 记 1.0，否则记 0.0。
      * S > 1：比「光速」还快。不是好消息，说明 T_sol 算错、或 kernel 偷工减料，
        交给 bench/check/integrity.py 判，不要在这里截断掩盖。
    """
    t_k = float(t_k_ms)
    t_b = float(t_b_ms)
    t_sol = float(t_sol_ms)
    for name, v in (("t_k_ms", t_k), ("t_b_ms", t_b), ("t_sol_ms", t_sol)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} 必须是正的有限数，实际 {v}")

    denom = t_b - t_sol
    if abs(denom) < _EPS:
        return 1.0 if t_k <= t_b else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / denom)


def from_ratio(r: float, t_b_ms: float, t_sol_ms: float) -> float:
    """已知加速比 r = T_b / T_k，反推可读分。

    用于「只记录了 ratio、没记 T_k」的历史数据；新数据请直接存 T_k。
    """
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"加速比必须为正的有限数，实际 {r}")
    t_b = float(t_b_ms)
    if not math.isfinite(t_b) or t_b <= 0:
        raise ValueError(f"t_b_ms 必须是正的有限数，实际 {t_b_ms}")
    return score(t_b / r, t_b, t_sol_ms)


def is_suspicious(s: float) -> bool:
    """S > 1 表示快过光速，须由 integrity 侧解释，不能当成绩报。"""
    return s > 1.0 + 1e-9
