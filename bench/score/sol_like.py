"""可读分：0.5 = 与基线持平，1 = 光速。

公式参考 SOL-ExecBench 的 sol_score，但**只用于报告，不用于奖励**——
它在接近光速处会饱和，用它比高下会掩盖差异。
"""


def score(t_k_ms: float, t_b_ms: float, t_sol_ms: float) -> float:
    """S = 1 / (1 + (T_k - T_sol) / (T_b - T_sol))。"""
    raise NotImplementedError


def from_ratio(r: float, t_b_ms: float, t_sol_ms: float) -> float:
    raise NotImplementedError
