#!/usr/bin/env python3
"""三臂性能对比报表（只认 results/bench_std.json）。

用法：
    python3 tools/report_std.py                 # 全部题
    python3 tools/report_std.py mla-sparse-l1   # 指定题

校验逻辑：
    1. bench_std.json 存在
    2. 每个 shape 的 axes 与 baselines/<task>.json 的对应 uuid **逐项相同**
       （多出的键忽略，缺键或值不同 → INVALID）
    3. 用 baseline 的 t_b_ms 算 ratio

不做这个校验就会出现"A 臂 SKV=16384 而 C/D 8192"还放在一起比的情况。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("A", "C", "D")


def attempt_dir(task: str, arm: str) -> str:
    if arm == "A":
        return os.path.join("/tmp/opbench-run", "arm-A", task, "attempt-1")
    return os.path.join(ROOT, "runs", f"arm-{arm}", task, "attempt-1")


def baseline(task: str) -> dict | None:
    p = os.path.join(ROOT, "baselines", f"{task}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_std(task: str, arm: str) -> tuple[dict | None, str]:
    d = attempt_dir(task, arm)
    p = os.path.join(d, "results", "bench_std.json")
    if not os.path.exists(p):
        return None, "no bench_std.json"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f), "ok"
    except Exception as e:
        return None, f"parse error: {e}"


def check_shapes(std: dict, base: dict) -> tuple[bool, str]:
    ref = {s.get("uuid"): (s.get("axes") or {}) for s in (base.get("shapes") or [])}
    got = {s.get("uuid"): (s.get("axes") or {}) for s in (std.get("shapes") or [])}
    missing = [u for u in ref if u not in got]
    if missing:
        return False, f"缺档位 {missing}"
    for uuid, ref_axes in ref.items():
        for k, v in ref_axes.items():
            if got[uuid].get(k) != v:
                return False, (f"{uuid}.{k}={got[uuid].get(k)} 与基准 {v} 不一致")
    return True, "ok"


def main(argv):
    tasks = argv or [t for t in sorted(os.listdir(os.path.join(ROOT, "tasks")))
                     if os.path.isdir(os.path.join(ROOT, "tasks", t))]
    for task in tasks:
        base = baseline(task)
        if not base:
            print(f"\n### {task}  (无 baseline，跳过)")
            continue
        tb = base.get("t_b_ms")
        tsol = base.get("t_sol_ms")
        cap = f"{tb/tsol:.2f}x" if tb and tsol else "?"
        print(f"\n### {task}    t_b={tb:.5f}ms  上限={cap}")
        print(f"{'arm':<5}{'状态':<18}{'median_ms':>11}{'min_ms':>10}{'ratio':>8}")
        for arm in ARMS:
            std, why = load_std(task, arm)
            if not std:
                print(f"{arm:<5}{why:<18}{'-':>11}{'-':>10}{'-':>8}")
                continue
            ok, msg = check_shapes(std, base)
            if not ok:
                print(f"{arm:<5}{'INVALID':<18}{'-':>11}{'-':>10}{'-':>8}   ← {msg}")
                continue
            # 关键：按 uuid 逐档匹配 baseline，ratio = base_median / arm_median。
            # 不能用 t_b_ms 做分母 —— 它是各档的几何平均，跟单一档位比会虚高。
            ref = {x.get("uuid"): x.get("median_ms") for x in (base.get("shapes") or [])}
            parts = []
            for r in std["shapes"]:
                u = r.get("uuid")
                arm_med = r.get("median_ms")
                base_med = ref.get(u)
                if not arm_med or not base_med:
                    continue
                parts.append((u, base_med, arm_med, base_med / arm_med))
            if not parts:
                print(f"{arm:<5}{'无匹配档位':<18}{'-':>11}{'-':>10}{'-':>8}")
                continue
            # 几何平均做汇总（与基线 t_b 的口径一致）
            import math
            agg = math.exp(sum(math.log(x[3]) for x in parts) / len(parts))
            detail = "  ".join(f"{u.split('-')[-1]}:{r:.2f}x" for u, _, _, r in parts)
            best = sorted(parts, key=lambda x: -x[3])[0]
            print(f"{arm:<5}{'ok':<18}{best[2]:>11.5f}{'-':>10}{agg:>7.2f}x   {detail}")


if __name__ == "__main__":
    main(sys.argv[1:])
