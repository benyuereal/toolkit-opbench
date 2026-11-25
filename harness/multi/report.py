"""汇总报告：把收集到的产物变成「题 × 臂 × 指标」的对照表。

## 三个维度，缺一不可

| 维度 | 权重 | 为什么必须有 |
|---|---|---|
| 正确性 | **硬门槛** | 不过就 0 分，多快都没用。业主原话：「正确性没有，写了也白搭」 |
| 性能 | 80% | 这是 bench |
| 证据/幻觉 | 判据 | 能区分「真做过」和「编了一堆数字」 |

## 性能用哪套数

**harness 复测的数才算数**。各臂自报的耗时会打架：实测同一份 lightop
实现被三臂分别报成 606 / 203 / 197 µs。自报数只用来判"有没有如实报告"。

所以这里区分 `rel_ratio`（本批最快 / 本臂，横向排名）与
`roof_ratio`（地板 / 本臂，离极限多远）。前者第一名永远 100%，
不告诉我们还有多少空间；后者能说出"这批实现都还差 40%"。

## 为什么允许结论是「没提高」

如果 C 和 A 打平，正确的输出是「skill 没带来提升」，不是去找一个更
好看的指标把它说成提升。`verdict()` 里 `uplift_pp` 为负或接近 0 时
明确写出来。
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt(v, nd=3, suffix=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def build_table(arts, scored: dict | None = None) -> dict:
    """arts: [multi.collect.Artifacts]，scored: {"task/arm": {...}}（可选）。

    返回 {"tasks": [...], "arms": [...], "rows": [...]}。
    """
    tasks = sorted({a.task for a in arts})
    arms = sorted({a.arm for a in arts})
    rows = []
    for a in sorted(arts, key=lambda x: (x.task, x.arm)):
        key = f"{a.task}/{a.arm}"
        s = (scored or {}).get(key, {})
        rows.append({
            "task": a.task,
            "arm": a.arm,
            "complete": a.complete,
            "has_report": a.has_report,
            "has_bench": bool(a.bench_std) and "_error" not in (a.bench_std or {}),
            "t_k_ms": s.get("t_k_ms"),
            "rel_ratio": s.get("rel_ratio"),
            "roof_ratio": s.get("roof_ratio"),
            "correct": s.get("correct"),
            "score": s.get("score"),
            "leaks": a.leaks,
        })
    return {"tasks": tasks, "arms": arms, "rows": rows}


def render(table: dict) -> str:
    tasks, arms, rows = table["tasks"], table["arms"], table["rows"]
    if not rows:
        return "(没有产物)"

    lines = ["=" * 92, "多臂对照", "=" * 92]
    for task in tasks:
        sub = [r for r in rows if r["task"] == task]
        lines.append(f"\n▸ {task}")
        lines.append(f"  {'臂':4s} {'产物':5s} {'t_k(ms)':>10s} {'rel':>7s} "
                     f"{'roof':>7s} {'正确':>5s} {'分':>5s}  泄漏")
        lines.append("  " + "-" * 80)
        for r in sub:
            lines.append(
                f"  {r['arm']:4s} {'✓' if r['complete'] else '✗':5s} "
                f"{_fmt(r['t_k_ms'], 4):>10s} "
                f"{_fmt(r['rel_ratio'], 3):>7s} "
                f"{_fmt(r['roof_ratio'], 3):>7s} "
                f"{('Y' if r['correct'] else ('N' if r['correct'] is False else '-')):>5s} "
                f"{_fmt(r['score'], 0):>5s}  "
                + (", ".join(r["leaks"]) if r["leaks"] else "-")
            )

        # 每题的相对差异：以最快者为 1.0
        vals = [(r["arm"], r["t_k_ms"]) for r in sub if r.get("t_k_ms")]
        if len(vals) >= 2:
            best = min(v for _, v in vals)
            spread = max(v for _, v in vals) / best if best else None
            lines.append(f"  最慢/最快 = {_fmt(spread, 2)}×"
                         f"（≥1.15 才有讨论价值；接近 1 说明这个维度上臂之间没有差异）")
    return "\n".join(lines)


def verdict(table: dict, baseline_arm: str = "A") -> str:
    """给出结论性判断。允许并鼓励「没有提高」。"""
    rows = table["rows"]
    tasks = [t for t in table["tasks"] if any(r["task"] == t and r.get("t_k_ms") for r in rows)]
    if not tasks:
        return "没有可比较的性能数据，无法判断。"

    deltas = []
    for task in tasks:
        sub = {r["arm"]: r for r in rows if r["task"] == task}
        base = sub.get(baseline_arm, {}).get("t_k_ms")
        if not base:
            continue
        for arm, r in sub.items():
            if arm == baseline_arm or not r.get("t_k_ms"):
                continue
            # 相对基线的提速（正 = 这个臂更快）
            improv = (base - r["t_k_ms"]) / base * 100
            deltas.append((task, arm, improv))

    if not deltas:
        return f"基线臂 {baseline_arm} 没有数据，无法计算差值。"

    lines = [f"相对 {baseline_arm} 臂的耗时变化（正 = 更快）："]
    for task, arm, d in sorted(deltas, key=lambda x: -x[2]):
        lines.append(f"  {task:32s} {arm}: {d:+.1f}%")

    mean = sum(d for _, _, d in deltas) / len(deltas)
    lines.append("")
    if abs(mean) < 5:
        lines.append(
            f"平均变化 {mean:+.1f}%，在噪声量级内 → **结论：没有可辨识的提升**。"
            "按业主的判据，这种情况应该削减文档而不是加更多题。"
        )
    elif mean > 0:
        lines.append(f"平均变化 {mean:+.1f}% → 有提升，但需逐题看是否集中在某一两道。")
    else:
        lines.append(f"平均变化 {mean:+.1f}% → 更慢。先查是不是预算/环境差异。")
    return "\n".join(lines)
