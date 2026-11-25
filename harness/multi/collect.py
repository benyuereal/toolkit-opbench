"""收集产物：把每个单元磁盘上的东西归拢成一份可比的表。

## 为什么"产物"和"结果"要分开

* **产物**（artifacts）：agent 写了什么文件。证明它动过手。
* **结果**（results）：里面有没有可复跑的数字。证明它做的事有意义。

只有产物没有结果 = 写了一堆散文；只有结果没有产物 = 数字来源无法追溯。
两个都要，但它们失败的含义不同，所以分开报。

## 为什么只用 harness 自己量的数排名

各臂自报的耗时不可比（实测同一份 lightop：606 / 203 / 197 µs）。
自报数只用来判"它有没有如实报告"，**排名一律用 `harness.measure` 复测**。
这里负责把两条都收上来，标清来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOLATED_ROOT = "/tmp/opbench-run"

# 这些文件名说明 agent 碰过"别处"的东西（跨臂偷看/读答案的痕迹）
LEAK_MARKERS = ("expect.md", "reference.py", "definition.json", "baselines/")


@dataclass
class Artifacts:
    task: str
    arm: str
    dir: str | None = None
    files: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    has_report: bool = False
    has_solution: bool = False
    bench_std: dict | None = None
    report_sha: str | None = None
    leaks: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def complete(self) -> bool:
        """kernel 题要有 bench_std.json；judgment 题要有 report.md。"""
        return bool(self.bench_std) or self.has_report


def _dirs(task: str, arm: str) -> list[str]:
    pats = [
        os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}", task, "attempt-*"),
        os.path.join(ROOT, "runs", f"arm-{arm.upper()}", task, "attempt-*"),
    ]
    out: list[str] = []
    for p in pats:
        out.extend(d for d in glob.glob(p) if os.path.isdir(d))
    return sorted(out)


def _pick_dir(dirs: list[str]) -> str | None:
    """从若干 attempt 目录里挑**真正有交付物**的那个。

    为什么不取编号最大的：`attempt-*` 这个名字不是 harness 独有的——
    实测 D 臂发现自己的工作目录被外部删掉后，**自己 mkdir 了一个
    attempt-2** 继续干活，结果只落在那里。它编号最大，但那不是
    harness 开的一轮。同理，harness 开的 attempt-1 可能是个空壳
    （模型说了句"我接下来要做 X"就退出，修复前会留空目录）。

    所以按"有没有 results/*.json 或 report.md"挑，都有就取最新的。
    都空的话返回最新的那个，让上层如实报"没有产物"。
    """
    if not dirs:
        return None
    with_deliverable = [
        d for d in dirs
        if glob.glob(os.path.join(d, "results", "*.json"))
        or os.path.isfile(os.path.join(d, "report.md"))
    ]
    pool = with_deliverable or dirs
    # 按 mtime 取最新，而不是按名字字典序：attempt-10 会排在 attempt-2 前面
    return max(pool, key=lambda d: os.path.getmtime(d))


def _read_bench_std(d: str) -> dict | None:
    p = os.path.join(d, "results", "bench_std.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def collect_one(task: str, arm: str) -> Artifacts:
    dirs = _dirs(task, arm)
    if not dirs:
        return Artifacts(task, arm, error="没有 attempt 目录")
    d = _pick_dir(dirs)
    a = Artifacts(task=task, arm=arm, dir=d)

    for root, _dirnames, files in os.walk(d):
        for n in files:
            rel = os.path.relpath(os.path.join(root, n), d)
            a.files.append(rel)

    rdir = os.path.join(d, "results")
    if os.path.isdir(rdir):
        a.results = sorted(os.path.relpath(p, d) for p in glob.glob(os.path.join(rdir, "*")))
    a.has_report = os.path.isfile(os.path.join(d, "report.md"))
    a.has_solution = os.path.isfile(os.path.join(d, "solution.json"))
    a.bench_std = _read_bench_std(d)

    rp = os.path.join(d, "report.md")
    if a.has_report:
        try:
            with open(rp, "rb") as f:
                a.report_sha = hashlib.sha256(f.read()).hexdigest()[:16]
        except OSError:
            pass

    # 泄漏痕迹：报告/脚本里提到答案侧文件
    for name in ("report.md",):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        for marker in LEAK_MARKERS:
            if marker in text:
                a.leaks.append(f"{name} 提到 {marker}")
    return a


def collect_all(units) -> list[Artifacts]:
    return [collect_one(u.task, u.arm) for u in units]


def render(arts: list[Artifacts]) -> str:
    """按题分组的产物对照表。"""
    by_task: dict[str, list[Artifacts]] = {}
    for a in arts:
        by_task.setdefault(a.task, []).append(a)

    lines = ["=" * 78, "产物收集", "=" * 78]
    for task, items in by_task.items():
        lines.append(f"\n▸ {task}")
        items = sorted(items, key=lambda x: x.arm)
        hdr = f"  {'臂':4s} {'结果':6s} {'报告':5s} {'results':8s} {'bench_std':10s} 泄漏"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for a in items:
            bs = "无"
            if a.bench_std:
                bs = "有" if "_error" not in a.bench_std else "坏"
            lines.append(
                f"  {a.arm:4s} {'✓' if a.complete else '✗':6s} "
                f"{'Y' if a.has_report else '-':5s} "
                f"{len(a.results):<8d} {bs:10s} "
                + (", ".join(a.leaks) if a.leaks else "-")
            )
            if a.error:
                lines.append(f"       ⚠ {a.error}")
    return "\n".join(lines)


def write(arts: list[Artifacts], path: str | None = None) -> str:
    out = path or os.path.join(ROOT, "reports", "multi-artifacts.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from dataclasses import asdict

    with open(out, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in arts], f, ensure_ascii=False, indent=2)
    return out
