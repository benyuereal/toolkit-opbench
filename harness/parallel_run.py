"""跨卡并行调度：不同题用不同卡并行，同一题的三臂用同一张卡。

    python3 -m harness.parallel_run --tasks norm-fused-add-rmsnorm-l1 moe-gating-l1

## 核心设计：可比性优先于并行度

我们要测的是"skill 的效应"（A/B/C 三臂的差）。这个差里**不能掺卡间差异**。
所以规则是：

  * **同一道题的三臂（A/B/C）必须用同一张卡**，且串行（它们之间
    本来就该严格同条件）。这是 Δ 干净的前提。
  * **不同题之间可以并行**，各用一张不同的卡。这是加速的来源。

于是并行度 = min(题目数, 空闲卡数)，而不是单元数。2 道题最多 2 路并行。

## 为什么不"每单元一张卡"（6 单元 6 卡）

那样同一题的 A 在卡0、B 在卡1、C 在卡2。三臂的 Δ 里就混进了
"卡0 vs 卡1 vs 卡2"的差异。本机 8 卡同型号同批次、sclk 一致，
卡间差异应该很小，但**我们测的就是效应，不该给它叠噪声**。
宁可用"题间并行"换 Δ 干净。

## 前提检查

  * 所有卡 sclk 档位一致（否则跨卡耗时不可比）——本机实测 8 卡都 1350 MHz。
  * 每单元用 CUDA_VISIBLE_DEVICES 绑卡 + OPBENCH_DEVICE 让 run.py 走绑卡模式
    （每卡一把锁；同卡上的三臂靠这把锁串行，不会叠）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available_cards(threshold: float = 20.0) -> list[int]:
    from bench.hw import dcu

    return dcu.free_devices(threshold)


def same_clock_level() -> tuple[bool, list[int]]:
    """所有卡 sclk 档位是否一致。"""
    from bench.hw import dcu

    levels = {lv for lv, _ in dcu._sclk_levels()}
    return len(levels) <= 1, sorted(levels)


def _run_task_on_card(task: str, arms, budget: int, max_usd: float, card: int):
    """一张卡上串行跑这道题的三臂（每卡锁保证同卡不叠，这里是顺序跑）。"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    env["OPBENCH_DEVICE"] = str(card)
    env["OPBENCH_ENV"] = os.environ.get("OPBENCH_ENV", "/workspace/tosc.env")
    env["OPBENCH_EFFORT"] = os.environ.get("OPBENCH_EFFORT", "medium")
    log = f"/tmp/parallel-{task}.log"
    with open(log, "w", encoding="utf-8") as lf:
        for arm in arms:
            print(f"    {task} / {arm} (卡{card})", flush=True)
            subprocess.call(
                [sys.executable, "-m", "harness.run", "--task", task, "--arm", arm,
                 "--budget", str(budget), "--max-usd", str(max_usd)],
                cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opbench-parallel-run")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--arms", default="A,B,C")
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=20.0)
    a = p.parse_args(argv)

    arms = tuple(x.strip().upper() for x in a.arms.split(",") if x.strip())
    ok, levels = same_clock_level()
    print(f"卡时钟档位: {levels} -> {'一致，可跨卡' if ok else '不一致，跨卡不可比！'}")
    if not ok:
        print("跨卡 Δ 不可信，等卡统一后重试。", file=sys.stderr)
        return 2

    free = available_cards(a.threshold)
    print(f"空闲卡（HCU% < {a.threshold}）: {free}")
    if not free:
        print("没有空闲卡，无法并行（claude-work 可能在用卡）。", file=sys.stderr)
        return 1

    n_par = min(len(a.tasks), len(free))
    print(f"并行度: {n_par} 题同时跑（每题 {len(arms)} 臂在同一张卡上串行）")

    threads = []
    for task, card in zip(a.tasks, free[:n_par]):
        t = threading.Thread(target=_run_task_on_card,
                             args=(task, arms, a.budget, a.max_usd, card),
                             name=f"task-{task}")
        t.start()
        threads.append(t)
        print(f"  启动 题={task} 卡={card}  thread={t.name}")
    # 题目数 > 空闲卡数时，多出的题排队（这里简化：只跑前 n_par 个）
    if len(a.tasks) > n_par:
        print(f"  注意: {len(a.tasks) - n_par} 题没卡，这次不跑")

    for t in threads:
        t.join()
        print(f"  完成 {t.name}")

    _print_status([(t, arm) for t in a.tasks for arm in arms])
    return 0


def _print_status(units):
    print("\n=== 各单元最终状态 ===")
    best = {}
    for f in glob.glob("/tmp/opbench-run/arm-*/*/attempt-*/meta.json") + \
             glob.glob(os.path.join(ROOT, "runs/arm-*/*/attempt-*/meta.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
            key = (d["arm"], d["task"])
            if key in units:
                best.setdefault(key, []).append(d)
        except Exception:
            pass
    for task, arm in units:
        cand = best.get((arm, task), [])
        if not cand:
            print(f"  {arm} {task:32s} (无结果)")
            continue
        d = max(cand, key=lambda x: str(x.get("started_at", "")))
        print(f"  {arm} {task:32s} {str(d.get('status')):10s} "
              f"wall={d.get('wall_time_s')}s card={d.get('device')} "
              f"art={len(d.get('artifacts') or [])}")


if __name__ == "__main__":
    raise SystemExit(main())
