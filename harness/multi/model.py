"""多臂实验的数据模型：题目口径 + 实验单元。

## 为什么要有这一层

一条臂的产物不能直接拿来互比——历史上踩过的三次错：

1. **各臂自选形状**。A 臂选 SKV=16384/TOPK=2048，C/D 用 8192/256，
   同一个"算子"耗时差 15 倍。那不是优化差异，是测的不是同一个算子。
2. **各臂自报口径不同**。A 交 JSON、C 用 `median_s`、D 用
   `rows[i].fused.median_ms`；同一个 lightop 实现三份数 606/203/197 µs。
3. **基线单位错**。swiglu 基线一度把 µs 当 ms 写进 `median_ms`，
   297.76 而不是 0.29776，达成率算出 0.5%。

所以口径必须在**下发之前**就定死，且每个单元都带着同一份口径跑。
`Caliber` 就是这个"定死的东西"，`from_baseline()` 是唯一的构造入口——
它只从 `baselines/<task>.json` 读，不许调用方传字面量。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class Caliber:
    """一道题的测量口径。同一道题的所有臂共用一个实例。"""

    task: str
    shapes: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]  # (uuid, ((axis,val),...))
    t_b_ms: float | None
    t_sol_ms: float | None
    peak_id: str | None
    peak_value: float | None
    work: str | None
    work_unit: str | None
    warmup: int
    iters: int
    flush_bytes: int

    @property
    def axes_list(self) -> list[dict]:
        return [dict(items) for _, items in self.shapes]

    def describe(self) -> str:
        """一行摘要，日志与报告里用。"""
        n = len(self.shapes)
        cap = (f" 上限 {self.t_b_ms / self.t_sol_ms:.2f}x"
               if self.t_b_ms and self.t_sol_ms else "")
        return (f"{self.task}: {n} 个形状, t_b={self.t_b_ms}, "
                f"t_sol={self.t_sol_ms}{cap}, peak={self.peak_id}")

    def digest(self) -> str:
        """口径指纹。三臂的口径不一致时必须能当场发现。"""
        import hashlib

        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def from_baseline(task_id: str, baselines_dir: str | None = None) -> Caliber | None:
    """从 baselines/<task>.json 构造口径。没有基线返回 None（judgment 题）。"""
    bdir = baselines_dir or os.path.join(ROOT, "baselines")
    path = os.path.join(bdir, f"{task_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    shapes = []
    for s in b.get("shapes") or []:
        axes = s.get("axes") or {}
        shapes.append((str(s.get("uuid", "?")),
                       tuple(sorted((k, int(v)) for k, v in axes.items()))))
    return Caliber(
        task=task_id,
        shapes=tuple(shapes),
        t_b_ms=b.get("t_b_ms"),
        t_sol_ms=b.get("t_sol_ms"),
        peak_id=b.get("peak_id"),
        peak_value=b.get("peak_value"),
        work=b.get("work"),
        work_unit=b.get("work_unit"),
        warmup=int(b.get("warmup") or 10),
        iters=int(b.get("iters") or 50),
        flush_bytes=int(b.get("flush_bytes") or 16 * 1024 * 1024),
    )


@dataclass
class Unit:
    """一个 (题, 臂) 组合。这是调度、进度、收集的最小粒度。"""

    task: str
    arm: str
    budget_s: int
    caliber: Caliber | None = None
    needs_gpu: bool = True
    tier: str | None = None

    # 运行时填充
    status: str = "pending"      # pending/running/ok/failed/skipped
    cwd: str | None = None
    attempt: int | None = None
    wall_s: float | None = None
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.task}/{self.arm}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["caliber"] = self.caliber.describe() if self.caliber else None
        d["caliber_digest"] = self.caliber.digest() if self.caliber else None
        return d


def build_units(tasks: list[dict], arms: list[str],
                budget_override: int | None = None,
                baselines_dir: str | None = None) -> list[Unit]:
    """把「题 × 臂」展开成单元清单。

    tasks: [{"id":..., "budget":..., "needs_gpu":..., "tier":...}]
    每道题只构造**一个** Caliber，所有臂共享同一实例——物理上不可能
    出现"这臂一个口径、那臂另一个口径"。
    """
    units: list[Unit] = []
    for t in tasks:
        task_id = t["id"]
        cal = from_baseline(task_id, baselines_dir)
        budget = budget_override or t.get("budget") or 1800
        for arm in arms:
            units.append(Unit(
                task=task_id,
                arm=arm,
                budget_s=budget,
                caliber=cal,
                needs_gpu=bool(t.get("needs_gpu", True)),
                tier=t.get("tier"),
            ))
    return units


def check_calibration(units: list[Unit]) -> list[str]:
    """同一道题的所有单元必须共享口径指纹。返回问题清单。

    这检查在物理上应该永远通过（build_units 只造一个实例），但保留它
    是因为单元可能来自不同批次（比如续跑时手工构造），而"两道题串了口径"
    这种错误一旦发生，结论会全错且很难看出来。
    """
    problems: list[str] = []
    by_task: dict[str, set[str]] = {}
    for u in units:
        if u.caliber is None:
            continue
        by_task.setdefault(u.task, set()).add(u.caliber.digest())
    for task, digests in by_task.items():
        if len(digests) > 1:
            problems.append(f"{task}: 同一题出现 {len(digests)} 套口径 {sorted(digests)}")
    return problems
