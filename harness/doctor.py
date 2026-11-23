"""体检：一条命令回答"这台机器现在能不能测"。

任一项红，都必须说清怎么修——不是只报错。

## 严重度

    ok    可以测
    warn  能测，但结论要降置信度（例如没锁频、当前在降频档）
    fail  不能测；继续跑只会产出不可比的数据

为什么要有 warn 这一档：本项目最容易犯的错不是"跑不了"，而是
"跑出来了但不可比"——600 MHz 下测的数、没锁频的数、另一会话在跑时的数，
看起来都正常，但拿它们算 ratio 就是在自欺。warn 让这类情况可见。

## 副作用

唯一会改系统状态的是 clock_lock 探针（save → manual+setsclk → load 还原）。
它在 **GPU 忙时会跳过**：改档位是全机的，另一个会话正在跑 bench 时
动时钟会污染它的测量。跳过时记 warn，不记 ok——没验过就是没验过。
"""
from __future__ import annotations

import json
import sys

CHECKS = [
    ("devices", "8 张卡可见"),
    ("event_timing", "torch.cuda.Event 可计时"),
    ("l2_size", "L2 = 8 MB"),
    ("sclk", "当前 sclk（空闲降频 / 工作档以实测为准）"),
    ("clock_lock", "能否锁频（hy-smi --save/--setperflevel/--setsclk/--load）"),
    ("libs", "lightop / aiter / deepgemm / torch / DTK 版本"),
    ("gpu_free", "另一会话是否占卡（HCU% <= 20）"),
    ("constants", "constants.yaml 三类峰值齐全"),
]

EXPECTED_DEVICES = 8
SEVERITY_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def _item(ok: bool, value, fix_hint: str = "", level: str | None = None) -> dict:
    if level is None:
        level = "ok" if ok else "fail"
    return {"ok": bool(ok), "level": level, "value": value, "fix_hint": fix_hint}


def _check_devices():
    from bench.hw import dcu

    n = dcu.device_count()
    if n == EXPECTED_DEVICES:
        return _item(True, f"{n} 张")
    return _item(
        False,
        f"{n} 张（期望 {EXPECTED_DEVICES}）",
        "容器没起对或驱动没加载：先 `hy-smi` 看卡在不在；"
        "不在就是 --device 映射少了，重启容器；"
        f"`hy-smi` 有卡但 torch 只见 {n} 张 → 检查 HIP_VISIBLE_DEVICES 是否被设过",
    )


def _check_event_timing():
    from bench.hw import dcu

    ok, note, per_iter = dcu.event_timing_works()
    if ok:
        return _item(True, f"可用（{note}）")
    return _item(
        False,
        note,
        "torch.cuda.Event 计时不可用意味着整套测量没有地基。"
        "先确认 `python3 -c \"import torch;print(torch.version.hip)\"` 非空（装成了 CUDA 版就废了）；"
        "再确认没用 pip 覆盖 DTK 自带的 torch（pip install torch 会打坏 HCC 后端）",
    )


def _check_l2():
    from bench.hw import constants as C
    from bench.hw import dcu

    info = dcu.query_devices()
    if info.count == 0:
        return _item(False, "无可用卡", "见 devices 一项：卡不可见时 L2 无从读起")
    try:
        expect = C.l2_bytes()
    except Exception as e:
        return _item(
            False, f"上游 L2 读不到：{e}",
            "constants.yaml 里 memory.l2_size 缺失或单位不是 MB；补上游，不要在代码里写死",
        )
    got = info.l2_bytes
    if got == expect:
        return _item(True, f"{got // 1024 // 1024} MB（与 constants 一致）")
    return _item(
        False,
        f"设备报 {got // 1024 // 1024} MB，constants 记 {expect // 1024 // 1024} MB",
        "不一致时**更新 constants**，不是改代码：flush buffer 必须 >= 2×L2，"
        "按小的那个取值会留下热数据（我们量出过 4571 GB/s 就是这么来的）",
    )


def _check_sclk():
    from bench.hw import dcu

    # 多次采样取众数：单次采样会把 auto DPM 的抖动读成"在降频"，
    # 每次体检都报一条假 warn。样本间隔足够短，不受另一会话的秒级波动影响。
    d = dcu.current_sclk_detail(samples=3)
    if d["min_mhz"] is None:
        return _item(
            False, "读不到",
            "hy-smi -g 没有输出：确认 `which hy-smi` 与驱动版本；"
            "读不到时钟就**不许**开始测量（测出来不知道是哪个档位）",
        )
    detail = (
        f"level {d['min_level']} = {d['min_mhz']} MHz"
        f"（各卡 {sorted(set(d['mhz']))}，perf={d['perf_level']}）"
    )
    if d["min_level"] >= dcu.WORK_LEVEL_MIN:
        return _item(True, detail)
    if d["min_level"] <= dcu.IDLE_LEVEL_MAX:
        return _item(
            False, detail + " ← 空闲降频档",
            "600 MHz 档下测出的耗时会偏大一半以上，且和 1350 MHz 的绝对值不可比。"
            "测量前必须先 ramp：跑 10–30s 饱和负载直到档位稳定到 "
            f"level {dcu.WORK_LEVEL_MIN}（{dcu.SCLK_LEVELS_MHZ[dcu.WORK_LEVEL_MIN]} MHz），"
            "再读 sclk 并把它写进 trace。",
            level="fail",
        )
    return _item(
        False, detail + " ← 未到工作档",
        f"当前不是上游标定峰值时的档位（level {dcu.WORK_LEVEL_MIN}）。"
        "要么 ramp 到位再测，要么在报告里注明档位并降置信度；"
        "别拿这个数直接和 1350 MHz 下的历史数据比",
        level="warn",
    )


def _check_clock_lock():
    from bench.hw import dcu

    busy = dcu._busy_pcts()
    if busy and max(busy) > dcu.BUSY_THRESHOLD:
        return _item(
            False,
            f"跳过（有卡 HCU%={max(busy):.1f}，另一会话在跑）",
            "锁频探针会改全机档位，忙时先不动。等 GPU 空闲后重跑 doctor；"
            "在此之前按 ramp + 记录 sclk 的方式测",
            level="warn",
        )
    ok = dcu.try_lock_clocks()
    if ok:
        return _item(
            True,
            "可以：save → setperflevel manual → setsclk → load --autorespond Y 还原",
        )
    return _item(
        False,
        "不行",
        "退路：不锁频，改为 ramp 到工作档 + 每次测量记录 sclk + 增加重复次数，"
        "并在报告里注明「未锁频」。三个坑已记在 dcu.try_lock_clocks 的 docstring："
        "--save 不覆盖已存在文件、setsclk 需先 manual、--load 需 --autorespond Y",
        level="warn",
    )


def _check_libs():
    from bench.hw import dcu

    libs = dcu.library_versions()
    required = ("lightop", "aiter", "deepgemm", "torch", "DTK")
    missing = [k for k in required if not libs.get(k)]
    shown = {k: libs.get(k) for k in required}
    if missing:
        return _item(
            False,
            {"versions": shown},
            f"缺 {missing} 的版本号。缺版本 = 结果不可复跑："
            "lightop/aiter/deepgemm 用 importlib.metadata 读不到，说明没装或包元数据坏了；"
            "DTK 看 /opt/dtk/.dtk_version",
        )
    return _item(True, shown)


def _check_gpu_free():
    from bench.hw import dcu

    pcts = dcu._busy_pcts()
    if not pcts:
        return _item(
            False, "读不到占用率",
            "hy-smi -u 无输出。读不到就按忙处理——宁可等，也不要和另一个会话抢卡",
            level="fail",
        )
    mx = max(pcts)
    busiest = [i for i, p in enumerate(pcts) if p > dcu.BUSY_THRESHOLD]
    if mx <= dcu.BUSY_THRESHOLD:
        return _item(True, f"空闲（max HCU%={mx:.1f}）")
    return _item(
        False,
        f"忙：卡 {busiest} HCU%={[round(pcts[i], 1) for i in busiest]}",
        "另一个会话（claude-work）可能在跑 bench。等它空下来再跑；"
        "自己的 bench 段必须包在 `bench.hw.dcu.gpu_lock()`（flock /tmp/opbench-gpu.lock）里，"
        "且开跑前再查一次 HCU%",
        level="fail",
    )


def _check_constants():
    from bench.hw import constants as C

    try:
        summary = C.summary()
    except Exception as e:
        return _item(
            False, f"读不到：{e}",
            f"上游 {C.UPSTREAM} 缺失或解析失败。峰值分母只有这一个来源，"
            "缺了就停——拿别处的数顶替正是我们栽过的两次（540、49.8）",
        )
    missing_kinds = [k for k, v in summary.items() if not v]
    if missing_kinds:
        return _item(
            False, summary,
            f"缺少类别 {missing_kinds}。三类的语义不可互换：报达成率用 peak_achievable，"
            "报实现好坏用 impl_measured，nominal 只能当上限",
        )
    unverified = [
        f"{k}:{p}" for k, v in summary.items() for p, d in v.items()
        if d.get("status") == "unverified"
    ]
    if unverified:
        return _item(
            True, summary,
            f"{unverified} 是 nominal/unverified：只能当上限引用，"
            "不能当达成率分母（docs/scoring.md 铁律 1）",
            level="warn",
        )
    return _item(True, summary)


_RUNNERS = {
    "devices": _check_devices,
    "event_timing": _check_event_timing,
    "l2_size": _check_l2,
    "sclk": _check_sclk,
    "clock_lock": _check_clock_lock,
    "libs": _check_libs,
    "gpu_free": _check_gpu_free,
    "constants": _check_constants,
}


def run() -> dict:
    """逐项跑，返回 {name: {ok, value, fix_hint}}。"""
    table: dict = {}
    for name, title in CHECKS:
        fn = _RUNNERS[name]
        try:
            item = fn()
        except Exception as e:  # 单项炸掉不该让整张体检表消失
            item = _item(
                False, f"{type(e).__name__}: {e}",
                "这是 doctor 自身的异常，先把这条贴给维护者；"
                "在这一项修好之前，体检结论不成立",
            )
        item["title"] = title
        table[name] = item
    return table


_SEV_MARK = {"ok": "✓", "warn": "!", "fail": "✗"}


def _fmt_value(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render(table: dict) -> str:
    lines = []
    lines.append("opbench doctor — 这台机器现在能不能测")
    lines.append("=" * 78)
    fails = warns = 0
    for name, title in CHECKS:
        item = table.get(name)
        if item is None:
            continue
        lvl = item.get("level") or ("ok" if item.get("ok") else "fail")
        if lvl == "fail":
            fails += 1
        elif lvl == "warn":
            warns += 1
        lines.append(f"[{_SEV_MARK[lvl]}] {name:<13} {title}")
        lines.append(f"      {_fmt_value(item.get('value'))}")
        hint = item.get("fix_hint")
        if hint and lvl != "ok":
            lines.append(f"      → {hint}")
    lines.append("-" * 78)
    verdict = (
        "可以测" if fails == 0 and warns == 0
        else ("可以测，但结论要降置信度（有 ! 项）" if fails == 0
              else "不能测：先修 ✗ 项")
    )
    lines.append(f"{len(table)} 项：{len(table) - fails - warns} ok / {warns} warn / {fails} fail ⇒ {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    """返回进程退出码：有 fail 才非 0；warn 不影响退出码（能测）。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    table = run()
    if "--json" in argv:
        print(json.dumps(table, ensure_ascii=False, indent=2))
    else:
        print(render(table))
    return 1 if any((i.get("level") or "ok") == "fail" for i in table.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
