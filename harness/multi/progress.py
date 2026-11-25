"""进度检测：不看"进程还在不在"，只看"产物到没到"。

## 为什么以产物为准

`claude -p` 的退出码不可靠地反映"做完了没有"（超时、被预算切断、
子进程残留都可能是 0）。有价值的事实是磁盘上有什么：

    attempt-N/meta.json          —— 这一轮跑过（含 status/wall_time）
    attempt-N/_stdout.jsonl      —— stdout 落盘（能判断是否被预算切断）
    attempt-N/report.md          —— 有结论
    attempt-N/results/*.json     —— 有原始数字

所以进度 = 对每个单元归类到这四种产物状态的组合。这样"卡住"和
"还在写"能分开：前者产物不动，后者 mtime 在变。

## 停在哪一步要能说清

只报"还在跑"没有信息量。这里给出每个单元的最后产物与 mtime，
谁停了 20 分钟不动一眼能看出来。
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOLATED_ROOT = "/tmp/opbench-run"

# 单个单元"多久没动就算卡住"
STALL_S = 900


@dataclass
class Snapshot:
    task: str
    arm: str
    attempt_dir: str | None
    has_meta: bool
    has_stdout: bool
    has_report: bool
    n_results: int
    n_artifacts: int
    last_mtime: float | None
    status: str | None
    wall_s: float | None

    @property
    def idle_s(self) -> float | None:
        return (time.time() - self.last_mtime) if self.last_mtime else None

    @property
    def done(self) -> bool:
        """有 report 或 results 才算产出。meta 只说明跑过。"""
        return self.has_report or self.n_results > 0

    @property
    def stalled(self) -> bool:
        return (not self.done) and (self.idle_s is not None) and self.idle_s > STALL_S

    def line(self) -> str:
        if not self.attempt_dir:
            return f"  {self.task:30s} {self.arm}  (未开始)"
        # idle 只在有产物时才显示时间；否则 "?" 比 "0min 未动" 更容易读错
        if self.idle_s is not None and self.done:
            idle = f"{self.idle_s/60:.0f}min 前更新"
        elif self.idle_s is not None:
            idle = f"{self.idle_s/60:.0f}min 未动"
        else:
            idle = "-"
        parts = [
            f"{self.task:30s} {self.arm}",
            f"status={self.status or '-':10s}",
            f"results={self.n_results}",
            f"report={'Y' if self.has_report else 'N'}",
            idle,
        ]
        if self.stalled:
            parts.append("⚠ 疑似卡住")
        elif self.done:
            parts.append("✓ 有产物")
        return "  " + "  ".join(parts)


def _attempt_dirs(task: str, arm: str) -> list[str]:
    """找这道题这条臂的所有 attempt 目录（隔离根 + 旧仓库根）。"""
    pats = [
        os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}", task, "attempt-*"),
        os.path.join(ROOT, "runs", f"arm-{arm.upper()}", task, "attempt-*"),
    ]
    out: list[str] = []
    for p in pats:
        out.extend(d for d in glob.glob(p) if os.path.isdir(d))
    return sorted(out)


def _meta(d: str) -> dict:
    p = os.path.join(d, "meta.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def snapshot(task: str, arm: str) -> Snapshot:
    dirs = _attempt_dirs(task, arm)
    if not dirs:
        return Snapshot(task, arm, None, False, False, False, 0, 0, None, None, None)
    d = dirs[-1]  # 只看最新一轮：续跑时旧 attempt 不该影响"现在到哪了"
    meta = _meta(d)
    results = glob.glob(os.path.join(d, "results", "*"))
    latest = 0.0
    for root, _dirs, files in os.walk(d):
        for n in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, n)))
            except OSError:
                pass
    return Snapshot(
        task=task,
        arm=arm,
        attempt_dir=d,
        has_meta=bool(meta),
        has_stdout=os.path.isfile(os.path.join(d, "_stdout.jsonl")),
        has_report=os.path.isfile(os.path.join(d, "report.md")),
        n_results=len(results),
        n_artifacts=len(meta.get("artifacts") or []),
        last_mtime=latest or None,
        status=meta.get("status"),
        wall_s=meta.get("wall_time_s"),
    )


def snapshot_all(units) -> list[Snapshot]:
    return [snapshot(u.task, u.arm) for u in units]


def render(snaps: list[Snapshot]) -> str:
    n_done = sum(1 for s in snaps if s.done)
    n_stall = sum(1 for s in snaps if s.stalled)
    head = (f"进度 {n_done}/{len(snaps)} 有产物"
            + (f"，{n_stall} 个疑似卡住" if n_stall else ""))
    return "\n".join([head, "-" * 74, *[s.line() for s in snaps]])
