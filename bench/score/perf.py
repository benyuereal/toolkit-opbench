"""性能评分：达成率（80% 权重）+ roofline 判据。

## 设计原则（业主定的）

1. **正确性是硬门槛**：不过 → 总分 0，多快都没用。
2. **性能占 80%**：这是 bench，不是别的东西。
3. **不框定在某个库**：达成率只看"跑多快"，不看"用哪个实现"。
   自己写的 kernel 跑得更快就该更高分。
4. **分母用三臂里最快的那一个**，不是我预先扫的值。

## 两个数，用途不同

* **相对达成率** = `本批最快耗时 / 该臂耗时`
  —— 用于横向排名。自写实现跑最快 → 它自己就是 100%。

* **roofline 达成率** = `理论地板 / 该臂耗时`
  —— 用于判"离物理极限还有多远"，回答"还有没有空间"。
  地板取 `max(带宽地板, launch 地板)`，因为小算子被 launch 支配。

**为什么两个都要**：相对达成率第一名永远是 100%（有人最快），
这不告诉我们离极限多远。roofline 达成率能说出"这批实现都还差 40%"。

## roofline 地板怎么算

    带宽地板 = work_bytes / peak_bandwidth
    launch 地板 = constants 的 launch.floor_us
    地板 = max(两者)

小算子（work < 几 MB）几乎总是被 launch 支配——这时再怎么优化 kernel 内部
都没用，该做的是融合或减少 launch。这个判断必须由地板算出来，不能拍脑袋。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerfScore:
    task: str
    arm: str
    t_ms: float
    rel_ratio: float          # 最快 / 本臂（横向排名用）
    roof_ratio: float         # 地板 / 本臂（离极限多远）
    floor_ms: float
    floor_kind: str           # "bandwidth" | "launch"
    bandwidth_floor_ms: float
    launch_floor_ms: float
    note: str = ""


def roofline_floor(defn, axes: dict) -> tuple[float, str, float, float]:
    """返回 (地板ms, 支配项, 带宽地板ms, launch地板ms)。

    work 表达式与单位从 definition.dcu 读（与 baseline.py 同口径），
    峰值从 constants 取——**不许写字面量**。
    """
    from bench.hw import constants as C

    dcu = defn.dcu or {}
    expr = dcu.get("work")
    if not expr:
        raise ValueError(f"{defn.id}: 缺 dcu.work，算不出 roofline 地板")
    unit = dcu.get("work_unit", "flop")
    peak_id = dcu.get("peak_constant")
    if not peak_id:
        raise ValueError(f"{defn.id}: 缺 dcu.peak_constant")

    work = eval(expr, {"__builtins__": {}}, dict(axes))
    peak = float(C.get(peak_id)["value"])

    if unit == "flop":
        bw_ms = work / (peak * 1e12) * 1e3
    elif unit == "byte":
        bw_ms = work / (peak * 1e9) * 1e3
    else:
        raise ValueError(f"dcu.work_unit 必须是 flop/byte，实际 {unit!r}")

    # launch 地板是全局常数（constants 的 launch.floor_us）
    try:
        launch_ms = float(C.get("launch.floor_us")["value"]) / 1e3
    except KeyError:
        launch_ms = 0.0

    if launch_ms >= bw_ms:
        return launch_ms, "launch", bw_ms, launch_ms
    return bw_ms, "bandwidth", bw_ms, launch_ms


def score_arm(defn, axes: dict, t_ms: float) -> PerfScore:
    """单个臂的 roofline 达成率（相对达成率要等所有臂齐了才算）。"""
    floor_ms, kind, bw, lf = roofline_floor(defn, axes)
    roof = floor_ms / t_ms if t_ms > 0 else 0.0
    note = ""
    if roof > 1.0:
        note = (f"快于 roofline 地板（{roof:.2f}×）——要么地板算低了，"
                "要么计时方式不同（如 burst 摊销 vs 逐次 event）。先核对计时口径")
    return PerfScore(
        task=defn.id, arm="", t_ms=t_ms, rel_ratio=0.0, roof_ratio=roof,
        floor_ms=floor_ms, floor_kind=kind,
        bandwidth_floor_ms=bw, launch_floor_ms=lf, note=note,
    )


def rank(times: dict[str, float], defn, axes: dict) -> dict[str, PerfScore]:
    """给一批臂算相对达成率 + roofline 达成率。

    `times` = {arm: t_ms}。最快的那一个 rel_ratio = 1.0。
    """
    if not times:
        return {}
    best = min(times.values())
    out = {}
    for arm, t in times.items():
        s = score_arm(defn, axes, t)
        s.arm = arm
        s.rel_ratio = best / t if t > 0 else 0.0
        out[arm] = s
    return out
