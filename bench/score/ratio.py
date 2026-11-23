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
            f"缺少基线 {path}。建档命令："
            f"python3 -m harness.baseline --task {task_id}"
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


# ------------------------------------------------------------------ CLI
#
# 退出码契约（与 verify 一致）：
#   0  全部可算
#   1  有题算不出（缺基线 / 缺结果）
#   2  用法错误
# 关键：**缺基线必须报错**，不许跳过。跳过会让"没算"看起来像"通过"。

def _load_results(attempt_dir: str) -> dict:
    import glob

    from bench.data import solution as sol_mod

    out = {}
    for p in sorted(glob.glob(os.path.join(attempt_dir, "results", "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                out[os.path.basename(p)] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    del sol_mod
    return out


def _first_timing(results: dict) -> float | None:
    """从结果文件里取被测实现的耗时（ms）。

    认两种写法：{"timing": {"median_ms": x}} 或顶层 "t_ms"/"median_ms"。
    取不到返回 None——不猜。
    """
    for obj in results.values():
        if not isinstance(obj, dict):
            continue
        t = obj.get("timing")
        if isinstance(t, dict) and isinstance(t.get("median_ms"), (int, float)):
            return float(t["median_ms"])
        for key in ("t_ms", "median_ms"):
            if isinstance(obj.get(key), (int, float)):
                return float(obj[key])
    return None


def score_run(run_dir: str, baselines_dir: str) -> tuple[dict, list[str]]:
    """对一个臂的运行目录算 ratio。返回 (结果表, 问题列表)。"""
    problems: list[str] = []
    rows: dict = {}
    if not os.path.isdir(run_dir):
        return rows, [f"运行目录不存在：{run_dir}"]

    tasks = sorted(
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d)) and not d.startswith("_")
    )
    if not tasks:
        return rows, [f"{run_dir} 下没有题目"]

    for task in tasks:
        attempt = os.path.join(run_dir, task, "attempt-1")
        if not os.path.isdir(attempt):
            problems.append(f"{task}: 没有 attempt-1")
            continue
        try:
            base = load_baseline(task, baselines_dir)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
            problems.append(f"{task}: 基线不可用（{e}）")
            continue
        results = _load_results(attempt)
        if not results:
            problems.append(f"{task}: 缺 results/*.json，无从算 ratio")
            continue
        t_k = _first_timing(results)
        if t_k is None:
            problems.append(f"{task}: results 里没有 timing/t_ms 字段")
            continue
        r = ratio(base["t_b_ms"], t_k)
        rows[task] = {
            "t_b_ms": base["t_b_ms"],
            "t_k_ms": t_k,
            "t_sol_ms": base["t_sol_ms"],
            "ratio": r,
        }
    return rows, problems


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="opbench-score")
    p.add_argument("--run", required=True, help="runs/arm-<X> 目录")
    p.add_argument("--baselines", default="baselines")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    try:
        rows, problems = score_run(a.run, a.baselines)
    except OSError as e:
        print(f"用法错误：{e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps({"run": a.run, "rows": rows,
                          "problems": problems}, ensure_ascii=False, indent=2))
    else:
        if rows:
            print(f"score：{a.run}")
            for task, r in rows.items():
                print(f"  {task:32s} T_b={r['t_b_ms']:.4f}ms  "
                      f"T_k={r['t_k_ms']:.4f}ms  ratio={r['ratio']:.3f}")
        for x in problems:
            print(f"  ✗ {x}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
