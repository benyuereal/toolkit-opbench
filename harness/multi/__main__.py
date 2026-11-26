"""多臂测试 CLI：预检 → 下发 → 进度 → 收集，四步可分开跑。

    python3 -m harness.multi --suite three-way --check-only   # 只预检
    python3 -m harness.multi --suite three-way                # 全跑
    python3 -m harness.multi --progress                       # 只看进度
    python3 -m harness.multi --collect                        # 只收集
    python3 -m harness.multi --quarantine                     # 收紧旧产物权限

## 为什么四步要能分开

跑一轮几小时。中途想知道"到哪了"、想单独重收集、想复跑一道题，
都不该被迫从头再来。分开之后每一步的输入都是磁盘上的事实，不依赖
上一次的进程状态。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARMS = "A,C,D"
DEFAULT_ENV = "/workspace/tosc.env"


def _load_state() -> dict | None:
    p = os.path.join(ROOT, "reports", "multi-state.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_state(units, path: str | None = None) -> str:
    out = path or os.path.join(ROOT, "reports", "multi-state.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([u.as_dict() for u in units], f, ensure_ascii=False, indent=2)
    return out


def _units_from_state(state: dict):
    from harness.multi.model import Caliber, Unit, from_baseline

    units = []
    for d in state:
        cal = from_baseline(d["task"])
        units.append(Unit(
            task=d["task"], arm=d["arm"], budget_s=int(d["budget_s"]),
            caliber=cal, needs_gpu=bool(d.get("needs_gpu", True)),
            tier=d.get("tier"),
        ))
    return units


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="opbench-multi",
        description="多臂对照实验（N 臂）：预检 / 下发 / 进度 / 收集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--suite", help="suites/<name>.yaml")
    p.add_argument("--tasks", nargs="+", help="题 id（与 --suite 二选一）")
    p.add_argument("--arms", default=DEFAULT_ARMS,
                   help=f"默认 {DEFAULT_ARMS}；可给任意多个，如 A,C,D,E")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--max-usd", type=float, default=None,
                   help="默认不设。设了会在题面时间预算之前切断")
    p.add_argument("--env", default=os.environ.get("OPBENCH_ENV", DEFAULT_ENV))
    p.add_argument("--card", type=int, nargs="+", default=None,
                   help="绑卡，可给多张（如 --card 4 5 6）。"
                        "串行时只用第一张；--parallel-cards 时按顺序分给各题。"
                        "默认自动挑空闲卡")
    p.add_argument("--parallel-cards", action="store_true",
                   help="题间并行（不同题不同卡），题内仍串行")
    p.add_argument("--threshold", type=float, default=20.0)
    p.add_argument("--check-only", action="store_true", help="只预检")
    p.add_argument("--progress", action="store_true", help="只报进度")
    p.add_argument("--collect", action="store_true", help="只收集产物")
    p.add_argument("--quarantine", action="store_true",
                   help="把旧产物树（早期未隔离产生的）收紧为 root 700")
    a = p.parse_args(argv)

    if a.quarantine:
        from harness.multi import isolation as iso

        done = iso.quarantine_legacy()
        print(f"已收紧 {len(done)} 个旧产物树：{done}")
        return 0

    # 同步改模块全局，否则本进程里的 _load_env() 会读旧的默认路径
    # （子进程靠继承 os.environ 没问题，但 preflight 是在本进程里查的）。
    from harness.launch import set_env_file
    set_env_file(os.path.abspath(a.env))
    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]

    # 只报进度 / 只收集：优先用上次的状态文件
    if a.progress or (a.collect and not (a.suite or a.tasks)):
        from harness.multi import collect as collect_mod
        from harness.multi import progress as prog_mod

        state = _load_state()
        if not state:
            print("没有 reports/multi-state.json，先跑一次完整流程", file=sys.stderr)
            return 2
        units = _units_from_state(state)
        if a.progress:
            print(prog_mod.render(prog_mod.snapshot_all(units)))
        if a.collect:
            arts = collect_mod.collect_all(units)
            print(collect_mod.render(arts))
            out = collect_mod.write(arts)
            print(f"\n产物清单：{os.path.relpath(out, ROOT)}")
        return 0

    # 建单元
    from harness.multi import model

    if a.tasks:
        tasks = [{"id": t} for t in a.tasks]
    elif a.suite:
        from harness.multi.suite import load_suite

        tasks = load_suite(a.suite)
    else:
        print("需要 --suite 或 --tasks（或 --progress / --collect）", file=sys.stderr)
        return 2
    if not tasks:
        print("没有选中任何题", file=sys.stderr)
        return 2

    units = model.build_units(tasks, arms, budget_override=a.budget)

    print("=" * 74)
    print("多臂对照 — 预检")
    print("=" * 74)
    print(f"题 {len(tasks)} 道，臂 {arms}，单元 {len(units)} 个")
    cals = {u.task: u.caliber for u in units if u.caliber}
    for task, cal in cals.items():
        print(f"  口径 {cal.describe()}")

    from harness.multi import preflight

    rep, free = preflight.run_all(units, a.env, a.threshold,
                                  parallel=a.parallel_cards)
    if rep.render():
        print(rep.render())
    if not rep.ok:
        print(f"\n✗ {len(rep.problems)} 项阻止开跑", file=sys.stderr)
        return 2
    print("✓ 预检通过")

    if a.check_only:
        print("\n(--check-only：未执行)")
        return 0

    # 下发
    from harness.multi import dispatch
    from harness.multi.model import Unit  # noqa: F401

    print("\n" + "=" * 74)
    print("下发")
    print("=" * 74)
    if a.parallel_cards:
        cards = a.card if a.card is not None else free
        print(f"题间并行，卡 {cards[:len(tasks)]}")
        dispatch.dispatch_by_card(units, cards, max_usd=a.max_usd)
    else:
        card = a.card[0] if a.card is not None else (free[0] if free else None)
        if card is None:
            print("没有可用卡", file=sys.stderr)
            return 2
        print(f"串行，绑卡 {card}（同一题的各臂共用这张卡）")
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
        os.environ["OPBENCH_DEVICE"] = str(card)
        try:
            dispatch.dispatch_sequential(units, max_usd=a.max_usd)
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    _save_state(units)

    # 进度 + 收集
    from harness.multi import collect as collect_mod
    from harness.multi import progress as prog_mod

    print("\n" + "=" * 74)
    print("进度")
    print("=" * 74)
    print(prog_mod.render(prog_mod.snapshot_all(units)))

    print("\n" + "=" * 74)
    print("产物")
    print("=" * 74)
    arts = collect_mod.collect_all(units)
    print(collect_mod.render(arts))
    out = collect_mod.write(arts)
    print(f"\n产物清单：{os.path.relpath(out, ROOT)}")
    print(f"状态：{os.path.relpath(_save_state(units), ROOT)}")

    bad = [u for u in units if u.status != "ok"]
    if bad:
        print(f"\n✗ {len(bad)} 个单元未成功：{[u.key for u in bad]}", file=sys.stderr)
        return 1
    print("\n✓ 全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
