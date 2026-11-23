"""主指标：对基线的加速比。不饱和，用于比较三臂。

为什么不用「达成率」当主指标：达成率的分母是问题（本项目在分母上错过两次，
一次估大 10.9×、一次估小 10.9×）。加速比只依赖两次实测，没有分母可错。
"""

from __future__ import annotations

import json
import math
import os


def ratio(t_b_ms: float, t_k_ms: float) -> float:
    """T_b / T_k。>1 表示比基线快。"""
    t_b = float(t_b_ms)
    t_k = float(t_k_ms)
    if not math.isfinite(t_b) or not math.isfinite(t_k):
        raise ValueError(f"耗时必须是有限数：t_b={t_b_ms}, t_k={t_k_ms}")
    if t_b <= 0 or t_k <= 0:
        raise ValueError(f"耗时必须为正：t_b={t_b_ms}, t_k={t_k_ms}")
    return t_b / t_k


def geomean_ratio(pairs: list[tuple[float, float]]) -> float:
    """跨形状几何平均，pairs 是 [(T_b, T_k), ...]。

    用几何平均而不是算术平均：算术平均会被单个超大加速比拉高，
    于是「挑一个好看的形状报数」就有收益。几何平均没有这个激励。
    """
    ratios = [ratio(b, k) for b, k in pairs]
    if not ratios:
        raise ValueError("pairs 为空，几何平均无定义")
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def load_baseline(task_id: str, baselines_dir: str) -> dict:
    """读取 baselines/<task_id>.json。

    基线缺失时报错而不是回落默认值——静默回落在本项目已经造成过事故：
    分母沿用了没有来源的 540，空间被估大 10.9×。
    """
    path = os.path.join(baselines_dir, f"{task_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少基线 {path}。先用参考实现建档："
            f"python3 -m harness.run --task {task_id} --baseline-write"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("t_b_ms", "t_sol_ms"):
        if key not in data:
            raise KeyError(f"{path} 缺 {key} 字段")
        if not isinstance(data[key], (int, float)) or isinstance(data[key], bool):
            raise TypeError(f"{path}.{key} 应为数值，实际是 {type(data[key]).__name__}")
    if data["t_b_ms"] <= 0:
        raise ValueError(f"{path}.t_b_ms 必须为正，实际 {data['t_b_ms']}")
    return data
