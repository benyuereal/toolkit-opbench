"""为一道题建档：跑出 T_b 与 T_sol，写 baselines/<id>.json。

## 为什么 T_b 不能用 torch 参考实现

torch 的 `A @ B` 是**正确性参考**，不是**性能基线**。拿未优化的实现当分母，
ratio 恒大于 1，什么实现都能"赢"——那证明不了任何事。

所以 T_b 必须来自**厂商已调优的实现**，且写清是哪个库、什么版本。
题面用 `definition.json` 的 `dcu.baseline_impl` 声明（点分路径），
这里按它 import 并计时；没有声明就报错，**不回落 torch**。

## 为什么 T_sol 从 constants 算

`T_sol = 工作量 / peak`，peak 只许引用 `bench.hw.constants` 的 id。
三类口径不许混：报达成率用 `peak_achievable`，报实现好坏用 `impl_measured`，
`peak_nominal` 只能当上限。

## 为什么要记 sclk 与 flush 配置

同一个 kernel 在 600 MHz 与 1350 MHz 上差一倍。不记档位的基线
半年后无法解释，也无法与今天的数据比较。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(ROOT, "baselines")


def _resolve_impl(dotted: str):
    """'lightop.fused_add_rms_norm' → 可调用对象。"""
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name:
        raise ValueError(f"baseline_impl 必须是点分路径，实际 {dotted!r}")
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, attr):
        raise AttributeError(f"{mod_name} 里没有 {attr}（题面声明的 baseline 实现不存在）")
    return getattr(mod, attr)


def _call_impl(fn, tensors: dict, defn) -> None:
    """把 build_inputs 造的张量按 definition.inputs 的键序喂进去。

    基线实现可能是「原地改写 + 返回元组」型（如 fused_add_rms_norm），
    也可能只返回一个张量。两种都接受，返回值忽略——我们只量时间。
    """
    args = [tensors[k] for k in defn.inputs]
    fn(*args)





def _workload_of(defn, td: str):
    from bench.data import workload as wl_mod

    path = os.path.join(td, "workload.jsonl")
    ws = wl_mod.load(path, defn)
    pub, _hold = wl_mod.split(ws)
    return pub


def _axes_for(w, defn) -> dict:
    axes = dict(defn.const_axes())
    axes.update(w.axes)
    return axes


def _t_sol_ms(defn, axes: dict, peak: float, unit: str) -> float:
    """按 definition 声明的工作量与口径算 T_sol。

    unit: "flop" → 2*M*N*K 这类；"byte" → 读+写字节数
    表达式里的名字从 axes 取，`eval` 限制在无内置的命名空间里
    （题面是仓库内文件，但仍不给 __builtins__）。
    """
    expr = (defn.dcu or {}).get("work")
    if not expr:
        raise ValueError(
            f"{defn.id}: dcu.work 未声明，无法算 T_sol。"
            "例如 gemm 写 \"2*M*N*K\"，rmsnorm 写 \"4*M*H*2\""
        )
    work = eval(expr, {"__builtins__": {}}, dict(axes))
    if unit == "flop":
        seconds = work / (peak * 1e12)
    elif unit == "byte":
        seconds = work / (peak * 1e9)
    else:
        raise ValueError(f"dcu.work_unit 必须是 flop 或 byte，实际 {unit!r}")
    return seconds * 1e3


def build(task_id: str, baselines_dir: str = BASELINES_DIR,
          warmup: int = 10, iters: int = 50, dry_run: bool = False) -> dict:
    from bench.data import definition as defn_mod
    from bench.hw import dcu
    from bench.timing import event
    from harness.run import task_dir

    td = task_dir(task_id)
    defn = defn_mod.load(os.path.join(td, "definition.json"))
    if defn.kind != "kernel":
        raise ValueError(f"{defn.id} 是 {defn.kind} 类，没有性能基线可建")

    dcu_cfg = defn.dcu or {}
    impl_path = dcu_cfg.get("baseline_impl")
    if not impl_path:
        raise ValueError(
            f"{defn.id}: definition.dcu.baseline_impl 未声明。"
            "T_b 必须来自厂商已调优实现，不许回落 torch（torch 是正确性参考，不是性能基线）"
        )
    peak_id = dcu_cfg.get("peak_constant")
    if not peak_id:
        raise ValueError(f"{defn.id}: definition.dcu.peak_constant 未声明")
    unit = dcu_cfg.get("work_unit", "flop")

    from bench.hw import constants as C

    peak_info = C.get(peak_id)
    peak = float(peak_info["value"])

    pub = _workload_of(defn, td)

    if dry_run:
        axes = _axes_for(pub[0], defn)
        return {
            "task": defn.id,
            "dry_run": True,
            "baseline_impl": impl_path,
            "peak_id": peak_id,
            "peak_value": peak,
            "t_sol_ms_first_shape": _t_sol_ms(defn, axes, peak, unit),
            "shapes": [_axes_for(w, defn) for w in pub],
        }

    impl = _resolve_impl(impl_path)

    timings = []
    sclk = None
    for w in pub:
        from bench.data import workload as wl_mod

        tensors = wl_mod.build_inputs(w, defn)
        t = event.time_call(lambda: _call_impl(impl, tensors, defn),
                            warmup=warmup, iters=iters,
                            name=f"{defn.id}@{w.uuid}")
        timings.append({
            "uuid": w.uuid,
            "axes": _axes_for(w, defn),
            "median_ms": t.median_ms,
            "mad_ms": t.mad_ms,
            "cv": t.cv,
            "sclk_mhz": t.sclk_mhz,
            "clock_stable": t.clock_stable,
        })
        sclk = t.sclk_mhz

    # T_b 取跨形状几何平均（防挑好看的形状）
    import math

    t_b = math.exp(sum(math.log(x["median_ms"]) for x in timings) / len(timings))
    # T_sol 同样跨形状几何平均
    sols = [_t_sol_ms(defn, x["axes"], peak, unit) for x in timings]
    t_sol = math.exp(sum(math.log(s) for s in sols) / len(sols)) if sols else None

    record = {
        "task": defn.id,
        "t_b_ms": t_b,
        "t_sol_ms": t_sol,
        "baseline_impl": impl_path,
        "peak_id": peak_id,
        "peak_value": peak,
        "peak_kind": peak_info["kind"],
        "peak_status": peak_info["status"],
        "work": dcu_cfg.get("work"),
        "work_unit": unit,
        "warmup": warmup,
        "iters": iters,
        "flush_bytes": 16 * 1024 * 1024,
        "sclk_mhz": sclk,
        "shapes": timings,
        "libs": dcu.library_versions(),
    }
    os.makedirs(baselines_dir, exist_ok=True)
    out = os.path.join(baselines_dir, f"{defn.id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    record["path"] = out
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opbench-baseline")
    p.add_argument("--task", required=True)
    p.add_argument("--baselines", default=BASELINES_DIR)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    try:
        rec = build(a.task, a.baselines, a.warmup, a.iters, a.dry_run)
    except (ValueError, AttributeError, FileNotFoundError, ImportError) as e:
        print(f"建档失败：{e}", file=sys.stderr)
        return 1
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
