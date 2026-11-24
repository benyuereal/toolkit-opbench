"""统一口径实测：不信三臂自报，自己量。

    python3 -m harness.measure --task norm-fused-add-rmsnorm-l1 \
        --impl lightop.fused_add_rms_norm --impl aiter.fused_add_rmsnorm_cu

## 为什么需要这个

实测发现三臂的 results 格式与计时口径**完全不可比**：

| 臂 | 字段 | 单位 | 口径 |
|---|---|---|---|
| A | 无 JSON（只有 .txt） | — | 手写日志 |
| B | median_s | 秒 | 有的带 flush 有的不带 |
| C | rows[i].fused.median_ms | 毫秒 | 逐次 event |

同一个 lightop 实现，各臂报出的数差 3 倍（B 说 606µs，C 说 203µs，
我统一测是 197µs）。**从它们的文件里拼排名是不可靠的**——拼出来的
差异可能全是口径差异，不是实现差异。

所以：**排名用我自己统一测的数**（本工具），各臂自报的数只作为
"它有没有如实报告"的证据（由 evidence 维度判）。

## 口径（全项目统一）

* 逐次 `torch.cuda.Event`，不用 burst 摊销（摊销会把 launch 藏掉）
* warmup 固定，iters 固定，报 median
* 每轮 flush L2（16MB）
* 原地改写的实现**轮换 buffer**（每轮喂干净输入）
* 记录 sclk

## 实现怎么给

`--impl 模块.函数`。参数按题目的 definition.inputs 键序喂张量，
`dcu.extra_args` 补标量，`dcu.inplace_inputs` 声明哪些输入会被改写。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(dotted: str):
    mod, _, attr = dotted.rpartition(".")
    if not mod:
        raise ValueError(f"必须是点分路径：{dotted!r}")
    m = importlib.import_module(mod)
    if not hasattr(m, attr):
        raise AttributeError(f"{mod} 里没有 {attr}")
    return getattr(m, attr)


def per_call_ms(fn, iters: int = 30, warmup: int = 6, flush_bytes: int | None = None):
    """逐次 event，返回 (median_ms, mad_ms, samples)。"""
    import torch

    from bench.timing import flush as flush_mod

    nb = flush_mod.DEFAULT_FLUSH_BYTES if flush_bytes is None else flush_bytes

    # 先把 Triton/编译的一次性开销付掉（实测首次调用可到秒级）
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()

    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for k in range(iters):
        flush_mod.flush(nb)
        ss[k].record()
        fn()
        ee[k].record()
    torch.cuda.synchronize()
    xs = [ss[k].elapsed_time(ee[k]) for k in range(iters)]
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return med, mad, xs


def measure(task_id: str, impl_path: str, shape_index: int = 0,
            iters: int = 30) -> dict:
    """测一个实现在一个 public 形状上的耗时。"""
    from bench.data import definition as defn_mod
    from bench.data import workload as wl_mod
    from bench.hw import dcu
    from harness.run import task_dir

    td = task_dir(task_id)
    defn = defn_mod.load(os.path.join(td, "definition.json"))
    ws = wl_mod.load(os.path.join(td, "workload.jsonl"), defn)
    pub, _ = wl_mod.split(ws)
    w = pub[shape_index]

    impl = resolve(impl_path)
    tensors = wl_mod.build_inputs(w, defn)
    extra = list((defn.dcu or {}).get("extra_args") or [])
    inplace = list((defn.dcu or {}).get("inplace_inputs") or [])

    # 原地改写：轮换 N 份 buffer，避免累加污染
    if inplace:
        N = 8
        bufs = [{**tensors}]
        for _ in range(N - 1):
            bufs.append({k: (v if k not in inplace else v.clone())
                         for k, v in tensors.items()})
        state = {"i": 0}

        def call():
            cur = bufs[state["i"] % N]
            state["i"] += 1
            impl(*([cur[k] for k in defn.inputs] + extra))
    else:
        def call():
            impl(*([tensors[k] for k in defn.inputs] + extra))

    sclk = dcu.current_sclk()
    med, mad, xs = per_call_ms(call, iters=iters)

    axes = dict(defn.const_axes()); axes.update(w.axes)
    return {
        "task": defn.id,
        "impl": impl_path,
        "shape": axes,
        "median_ms": med,
        "mad_ms": mad,
        "cv": mad / med if med else None,
        "samples": [round(x, 5) for x in xs],
        "sclk_mhz": sclk,
        "iters": iters,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opbench-measure")
    p.add_argument("--task", required=True)
    p.add_argument("--impl", action="append", required=True,
                   help="模块.函数，可给多次")
    p.add_argument("--shape-index", type=int, default=0)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    rows = []
    for impl in a.impl:
        try:
            r = measure(a.task, impl, a.shape_index, a.iters)
            rows.append(r)
            print(f"{impl:40s} {r['median_ms']:.5f} ms  mad={r['mad_ms']:.5f}  "
                  f"cv={r['cv']:.3f}  sclk={r['sclk_mhz']}", flush=True)
        except Exception as e:
            print(f"{impl:40s} FAILED {type(e).__name__}: {e}", file=sys.stderr)

    if rows:
        best = min(r["median_ms"] for r in rows)
        print(f"\n最快 {best:.5f} ms；相对达成率：")
        for r in rows:
            print(f"  {r['impl']:40s} {best / r['median_ms'] * 100:6.1f}%")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
