"""下发任务：把一个 Unit 变成一个受隔离约束的 agent 进程。

## 为什么不让调用方自己拼命令

三臂对照的全部意义是"**只有上下文不同**"。任何别的差异都会伪装成结论：

  * 不同预算 → 测出来是预算差
  * 不同模型 → 测出来是模型差
  * 不同 allowedTools → 测出来是工具权限差
  * 不同 prompt → 测出来是提示差

所以命令构造集中在 `harness.launch.build_cmd`，这里只负责：
  1. 检查这道题这条臂**读不到答案**；
  2. 检查已有臂的预算与本轮一致（`assert_arms_comparable`）；
  3. 把口径块注入 prompt（三臂逐字相同）；
  4. 落到该臂独立用户的 cwd 里跑。

## 串行 vs 并行

同一道题的各臂**必须同卡**（Δ 里不能混卡间差异），所以题内串行。
不同题之间可以并行用不同卡。`dispatch_sequential` 是保守默认；
`dispatch_by_card` 把题分配到卡上并行。

## 失败不静默

任何异常都记在该 Unit 上并继续下一个——一轮跑几小时，一个臂挂掉不该
让剩下的单元不跑。但最后汇总会把这些单元列出来并返回非零。
"""
from __future__ import annotations

import os
import time
from typing import Iterable


def dispatch_one(unit, max_usd: float | None = None,
                 card: int | None = None) -> object:
    """跑一个单元。异常写进 unit.error，不抛出。

    `card` 显式传给 `run_one`，**不写进程级环境变量**：多线程并行时
    `os.environ` 是共享的，卡 5 的线程刚设成 "5"、卡 4 的线程下一行就
    改成 "4"，子进程绑哪张卡全看时序，而且不会报错——只会表现为
    "某张卡上的题没产出"。
    """
    from harness.launch import run_one

    unit.status = "running"
    started = time.time()
    try:
        r = run_one(unit.task, unit.arm, unit.budget_s, max_usd=max_usd,
                    device=card)
    except Exception as e:
        unit.status = "failed"
        unit.error = f"{type(e).__name__}: {e}"
        unit.wall_s = round(time.time() - started, 1)
        return unit

    unit.wall_s = r.get("wall_time_s") or round(time.time() - started, 1)
    unit.status = "ok" if str(r.get("status")) in ("ok", "completed", "success") else str(r.get("status"))
    unit.attempt = r.get("attempt")
    unit.artifacts = list(r.get("artifacts") or [])
    unit.cwd = r.get("cwd")
    if unit.status != "ok":
        unit.error = r.get("error") or r.get("terminal_reason")
    return unit


def dispatch_sequential(units: Iterable, max_usd: float | None = None,
                        on_done=None) -> list:
    """按顺序跑，每单元内三臂连续（同题同卡，Δ 干净）。"""
    out = []
    for u in units:
        print(f"  → {u.task} / {u.arm}  ({u.budget_s}s)", flush=True)
        dispatch_one(u, max_usd=max_usd)
        flag = "✓" if u.status == "ok" else "✗"
        print(f"    {flag} status={u.status} wall={u.wall_s}s "
              f"artifacts={len(u.artifacts)}"
              + (f"  {u.error}" if u.error else ""), flush=True)
        if on_done:
            on_done(u)
        out.append(u)
    return out


def dispatch_by_card(units: Iterable, cards: list[int],
                     max_usd: float | None = None) -> list:
    """题间并行：一道题分一张卡，题内各臂串行。

    并行度 = min(题数, 卡数)。不按"单元一张卡"是因为同一题的臂换卡会
    把卡间差异混进 Δ——我们测的是 skill 的效应，不该给它叠噪声。
    """
    import threading

    units = list(units)
    by_task: dict[str, list] = {}
    for u in units:
        by_task.setdefault(u.task, []).append(u)

    results: list = []
    lock = threading.Lock()
    queue = list(by_task.items())

    def worker(card: int):
        while True:
            with lock:
                if not queue:
                    return
                task, items = queue.pop(0)
            for u in items:
                print(f"  → [卡{card}] {u.task} / {u.arm}", flush=True)
                try:
                    dispatch_one(u, max_usd=max_usd, card=card)
                except Exception as e:
                    # 一道题炸掉不能让整张卡静默退场。
                    #
                    # 实测（2026-11-26 10:05）：三题并行，卡 5/6 的线程
                    # 各抛了一次异常就没了 —— `worker()` 里当时没有 except，
                    # 线程死掉后 `join()` 照常返回，主进程以为跑完了。
                    # 结果是三题里只有一题在跑，而日志里一个字都没有。
                    u.status = f"error:{type(e).__name__}"
                    print(f"    [卡{card}] {u.arm} {u.task} 失败："
                          f"{type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                else:
                    print(f"    [卡{card}] {u.arm} {u.status} "
                          f"wall={u.wall_s}s", flush=True)
                with lock:
                    results.append(u)

    threads = [threading.Thread(target=worker, args=(c,), name=f"card-{c}")
               for c in cards[:len(by_task)]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
