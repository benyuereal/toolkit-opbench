"""主指标：对基线的加速比。不饱和，用于比较三臂。"""


def ratio(t_b_ms: float, t_k_ms: float) -> float:
    """T_b / T_k。>1 表示比基线快。"""
    raise NotImplementedError


def geomean_ratio(pairs: list[tuple[float, float]]) -> float:
    """跨形状几何平均，pairs 是 [(T_b, T_k), ...]。"""
    raise NotImplementedError


def load_baseline(task_id: str, baselines_dir: str) -> dict:
    raise NotImplementedError
