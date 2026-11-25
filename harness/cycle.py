"""完整周期：一道题从「没有基线」到「三臂 Δ」的一条命令。

    python3 -m harness.cycle --tasks gemm-bf16-l1 norm-fused-add-rmsnorm-l1 moe-gating-l1

## 为什么需要它

六步工作流（doctor → prepare → run → verify → score → report）已经各自能跑，
但一步接一步靠人记：先建基线、再跑三臂、再 verify、再 score、最后 collect。
漏掉任何一步都会得到"看起来正常但不可比"的结果——这正是我们已经踩过的坑。

这个脚本把顺序固化下来，并在每一步**失败即停**：
基线建不成，后面的三臂跑了也是白跑。

## 与 make 的关系

`make run SUITE=x` 是"按 suite 跑一个臂"；这里是"把几道题走完全程"。
两者互补：前者用于按套件批量跑，后者用于验证"这套题真的能用"。

## GPU 串行

所有需要 GPU 的步骤都在 `gpu_lock()` 里，且每步开跑前重新挑空闲卡。
不会与前一个会话或另一个 agent 互相污染。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("A", "B", "C")


def _step(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def _needs_gpu(task_id: str) -> bool:
    from harness.run import load_definition, needs_gpu, task_dir

    return needs_gpu(load_definition(task_dir(task_id)))


def ensure_baseline(task_id: str, iters: int = 30) -> dict:
    """没有基线就建。已有则跳过——基线是**一次性存档**，
    每次跑都重建会让不同轮次的三臂对照失去共同参照。"""
    path = os.path.join(ROOT, "baselines", f"{task_id}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    from bench.hw import dcu
    from harness import baseline

    with dcu.gpu_lock(wait_s=900, poll_s=10) as lock:
        if not lock["acquired"]:
            raise RuntimeError(f"等不到 GPU 锁（{lock.get('waited_s')}s）")
        dev = dcu.pick_device()
        if dev is None:
            raise RuntimeError(f"没有空闲卡：{dcu._busy_pcts()}")
        rec = baseline.build(task_id, warmup=10, iters=iters)
        rec["device"] = dev
    return rec


def run_arms(task_id: str, budget: int, max_usd: float) -> list[dict]:
    from harness.run import run_one

    out = []
    for arm in ARMS:
        print(f"  → {task_id} / {arm}", flush=True)
        r = run_one(task_id, arm, budget, max_usd=max_usd)
        print(f"     status={r.get('status')} wall={r.get('wall_time_s')}s "
              f"artifacts={len(r.get('artifacts') or [])}", flush=True)
        out.append(r)
    return out


def verify_arm(arm: str) -> list[str]:
    from bench.check import integrity

    run_dir = os.path.join(ROOT, "runs", f"arm-{arm}")
    if not os.path.isdir(run_dir):
        return []
    return integrity.verify_run(run_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opbench-cycle")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--budget", type=int, default=None,
                   help="默认用题面自己写的时间预算")
    p.add_argument("--max-usd", type=float, default=None,
                   help="默认不设美元上限，让臂跑满题面写的时间预算。"
                        "设了会和题面的 wall-time 预算互相打架——"
                        "实测 $5 会在 ~33 分钟切断，60 分钟的题永远跑不完，"
                        "而且三臂会被切在近似但不相同的时刻，Δ 被污染。")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--arms", default="A,B,C")
    a = p.parse_args(argv)

    global ARMS
    ARMS = tuple(x.strip().upper() for x in a.arms.split(",") if x.strip())

    from harness.run import load_definition, stated_budget_s, task_dir

    summary = {}
    for task_id in a.tasks:
        _step(f"题目 {task_id}")
        td = task_dir(task_id)
        defn = load_definition(td)
        task_id = defn.id

        if defn.kind == "kernel" and not a.skip_baseline:
            try:
                rec = ensure_baseline(task_id)
                print(f"  基线: t_b={rec['t_b_ms']:.4f} ms  "
                      f"t_sol={rec['t_sol_ms']:.4f} ms  "
                      f"impl={rec.get('baseline_impl')}", flush=True)
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                print(f"  ✗ 基线建不成：{e}", file=sys.stderr)
                summary[task_id] = {"baseline": f"failed: {e}"}
                continue
        elif defn.kind == "judgment":
            print("  （judgment 类，无性能基线）", flush=True)

        with open(os.path.join(td, "task.md"), encoding="utf-8") as f:
            stated = stated_budget_s(f.read())
        budget = a.budget or stated or 1800

        results = run_arms(task_id, budget, a.max_usd)
        summary[task_id] = {
            "budget": budget,
            "arms": {r["arm"]: r.get("status") for r in results},
        }

    _step("校验")
    for arm in ARMS:
        problems = verify_arm(arm)
        if problems:
            print(f"  arm-{arm}: {len(problems)} 个问题")
            for x in problems[:5]:
                print(f"    ✗ {x}")
        else:
            print(f"  arm-{arm}: 通过")

    _step("汇总")
    from harness import collect as collect_mod

    data = collect_mod.collect(os.path.join(ROOT, "runs"))
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    path = collect_mod.write_report(os.path.join(ROOT, "runs"),
                                    os.path.join(ROOT, "reports"))
    print(collect_mod.render_report(data))
    print(f"\n报告：{os.path.relpath(path, ROOT)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
