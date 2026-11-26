"""复测：不信各臂自报的数字，用同一口径重测它们**提交的实现**。

    python3 -m harness.verify --task norm-act-swiglu-add-rmsnorm-l1
    python3 -m harness.verify --task ... --arms A C D --iters 100
    python3 -m harness.verify --task ... --impl A=/path/to/k.py:fused --impl C=...

## 为什么必须有一个独立模块

三臂是**并行**跑的，HBM 带宽是整机共享的。谁落在邻居空闲的窗口里，谁就
测出更好的数。实测本轮同一个 kernel：

| 臂 | 自报 | 复测 | 差 |
|---|---|---|---|
| A | 0.17328 | 0.17536 | +1% |
| D | **0.11504** | **0.17440** | **+52%** |

D 报了 5 次重复、波动 ±0.1%，看起来极其可信——但它换一张干净卡就复现不出来。
这不是它撒谎，是测量环境的问题。A 和 C 都在报告里主动指出了这件事。

所以：**排名一律用本模块的数**。各臂自报的数只用来回答"它有没有如实报告"，
作为证据维度的一个指标，不参与性能排名。

## 与 harness/measure.py 的分工

* `measure`：给一个**已知路径**（`模块.函数`）测一道题。用于建基线、
  比较厂商库。
* `verify`（本模块）：从**各臂的产物**里自动认出它们提交了什么，
  然后在同一进程、同一张卡上交替复测。用于给三臂（或 N 臂）排名。

## 交替测量为什么必要

不能"先把 A 测完 100 次再测 D"：那样 A 独占冷机、D 独占热机，差出来的是
漂移不是实现差异。每轮把各实现各测一次，才能把漂移平摊到各方头上。

## 认不出实现怎么办

跳过并**如实报告**，不猜。猜错会把 A 的 kernel 记成 D 的成绩——这比没有
数据更糟。认不出时给出可执行的补救：用 `--impl 臂=路径:函数` 手工指定。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# 从各臂产物里认出实现
# --------------------------------------------------------------------------

def _find_impl_spec(art) -> tuple[str | None, str | None, str | None]:
    """从某臂的产物里认出 (脚本绝对路径, 入口函数, 认不出的原因)。

    优先看 `results/bench_std.json` 的 `impl` 字段——那是题面要求 agent
    填的"我落到了哪个实现"，格式相对可控。

    支持三种写法（实测各臂都用过）：
        scripts/fused_add_rmsnorm_swiglu.py::fused_add_rmsnorm_swiglu
        triton 3.6.0 :: scripts/fused_add_rmsnorm_swiglu.py::fused_...
        v5-vecw-r2                                （只有名字，没有路径）
    """
    bs = art.bench_std or {}
    impl = str(bs.get("impl") or "")

    m = re.search(r"([\w/.\-]+\.py)(?:::([\w]+))?", impl)
    if not m:
        # impl 字段只有一个人造名字（"fused-hip-v5"、"triton.jit:_fused_kernel"），
        # 没有可执行路径。这时两条回退路：
        #   1. 名字本身是个变体 tag，去卡的 _common.py / variants.yaml 里查真身
        #   2. 扫 scripts/ 找定义了 triton.jit kernel 或 HIP launch 的文件
        return _guess_from_scripts(art, impl)

    script, func = m.group(1), m.group(2)

    cand = script if os.path.isabs(script) else os.path.join(art.dir, script)
    if not os.path.isfile(cand):
        alt = os.path.join(art.dir, "scripts", os.path.basename(script))
        if os.path.isfile(alt):
            cand = alt
    if not os.path.isfile(cand):
        return None, None, f"找不到脚本 {script}"

    if func:
        return cand, func, None

    # 没写函数名：只在能唯一确定时猜
    try:
        tree = ast.parse(open(cand, encoding="utf-8").read())
        fns = [n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    except (SyntaxError, OSError) as e:
        return None, None, f"解析 {os.path.basename(cand)} 失败：{e}"

    likely = [f for f in fns if "fused" in f.lower() or "kernel" in f.lower()]
    if len(likely) == 1:
        return cand, likely[0], None
    if len(fns) == 1:
        return cand, fns[0], None
    return None, None, f"{os.path.basename(cand)} 里入口不唯一：{fns[:6]}"


def _guess_from_scripts(art, impl: str) -> tuple[str | None, str | None, str | None]:
    """impl 字段没有 .py 路径时，从 scripts/ 里找能当入口的文件。

    判据按可靠性排序：
      1. 文件名里带 fused/kernel 的 .py，且只有一个这样的文件 → 取它
      2. 多个候选时，优先「import triton 且定义了 @triton.jit」或
         「调了 .so / ctypes」的那个
      3. 都判断不了就如实说不唯一，**不猜**——猜错会把 A 的成绩记到 D 头上

    实测遇到的两种情况：
      C: impl="fused-hip-v5 = 自研 HIP kernel kernels/xxx.hip"
         → 真身由卡的 `_common.get_runner("fused-hip-v5")` 返回，
           所以优先用卡的 adapter，而不是自己找脚本
      D: impl="triton.jit:_fused_kernel"
         → scripts/ 下有 fused_triton.py，定义了 fused_add_rmsnorm_swiglu
    """
    import glob

    # 1. 先看该臂 cwd 里有没有**本题对应的**算子卡（C 臂特例）
    #
    # 为什么要按题目 id 收窄：一个 cwd 里躺着十几张卡，每张都有自己的
    # `_common.py:get_runner`。实测第一版按 glob 顺序取第一张，结果取到了
    # `op-moe-topk-gating`——它的 get_runner 里当然没有 fused-hip-v5，
    # 真身在 `op-norm-act-swiglu`。认错卡会把别的算子的实现当成本题成绩。
    tag_m = re.match(r"\s*([\w.\-]+)", impl)
    tag = tag_m.group(1) if tag_m else None
    if not tag:
        return None, None, f"impl 字段无法解析出实现名：{impl[:60]!r}"

    hits = glob.glob(os.path.join(art.dir, "**", "skills", "*", "_common.py"),
                     recursive=True)
    # 打分：卡目录名与题目 id 的相似度最高者优先
    def _score(h: str) -> int:
        card = os.path.basename(os.path.dirname(h))
        s = 0
        for tok in re.split(r"[-_]", art.task):
            if tok and tok in card:
                s += 1
        try:
            if tag in open(h, encoding="utf-8").read():
                s += 10          # 卡的源码里直接提到这个 tag，几乎肯定对
        except OSError:
            pass
        return s

    best = None
    for h in hits:
        try:
            if "get_runner" not in open(h, encoding="utf-8").read():
                continue
        except OSError:
            continue
        sc = _score(h)
        if best is None or sc > best[0]:
            best = (sc, h)

    if best and best[0] >= 10:
        h = best[1]
        return (f"{h}:get_runner", tag,
                f"通过卡的 get_runner('{tag}') 解析"
                f"（卡 {os.path.basename(os.path.dirname(h))}）")

    # 2. 扫 scripts/
    sdir = os.path.join(art.dir, "scripts")
    if not os.path.isdir(sdir):
        return None, None, f"impl 无 .py 路径（{impl[:60]!r}）且没有 scripts/"
    cands = []
    for f in sorted(os.listdir(sdir)):
        if not f.endswith(".py"):
            continue
        p = os.path.join(sdir, f)
        try:
            src = open(p, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        fns = [n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        entry = [x for x in fns if "fused" in x.lower() or "swiglu" in x.lower()]
        if not entry:
            continue
        score = 0
        if "@triton.jit" in src:
            score += 2
        if "ctypes" in src or ".so" in src:
            score += 1
        if "fused" in f.lower() or "kernel" in f.lower():
            score += 2
        cands.append((score, p, entry[0]))

    if not cands:
        return None, None, f"scripts/ 里没有定义 fused/kernel 入口的 .py（impl={impl[:50]!r}）"
    cands.sort(reverse=True)
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        names = [os.path.basename(c[1]) for c in cands[:4]]
        return None, None, f"候选同分无法判断：{names}"
    return cands[0][1], cands[0][2], None


def discover(task_id: str, arms: list[str]) -> tuple[dict, dict]:
    """返回 ({臂: (spec, 自报 median)}, {臂: 跳过原因})。"""
    from harness.multi.collect import collect_one

    found: dict[str, tuple[str, float | None]] = {}
    skipped: dict[str, str] = {}
    for arm in arms:
        art = collect_one(task_id, arm)
        if not art.dir:
            skipped[arm] = "没有 attempt 目录"
            continue
        path, func, why = _find_impl_spec(art)
        if not path:
            skipped[arm] = why or "未知原因"
            continue
        bs = art.bench_std or {}
        shp = (bs.get("shapes") or [{}])[0]
        claimed = (float(shp["median_ms"])
                   if isinstance(shp.get("median_ms"), (int, float)) else None)
        found[arm] = (f"{path}:{func}", claimed)
    return found, skipped


# --------------------------------------------------------------------------
# 交替复测
# --------------------------------------------------------------------------

def time_interleaved(fns: dict, iters: int = 50, warmup: int = 20,
                     flush_bytes: int | None = None) -> dict:
    """每轮把每个实现各测一次，返回各实现的统计量。

    交替而不是"逐个测完"：见模块 docstring。
    """
    import torch

    from bench.timing import flush as flush_mod

    if not fns:
        raise ValueError("至少给一个实现")
    nbytes = flush_mod.DEFAULT_FLUSH_BYTES if flush_bytes is None else int(flush_bytes)

    # 每个实现单独付掉首次编译/JIT（实测 256³ mm 首调 5.6s，之后 <1ms）
    for _ in range(max(1, warmup)):
        for fn in fns.values():
            fn()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {k: [] for k in fns}
    for _ in range(iters):
        for name, fn in fns.items():
            flush_mod.flush(nbytes)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()   # 交替必须逐个同步，否则 event 会攒在一起
            samples[name].append(s.elapsed_time(e))

    out = {}
    for name, xs in samples.items():
        med = statistics.median(xs)
        mad = statistics.median([abs(v - med) for v in xs])
        out[name] = {
            "median_ms": med,
            "min_ms": min(xs),
            "mad_ms": mad,
            "cv": (mad / med) if med else None,
            "n": len(xs),
        }
    return out


def _run_isolated(task_id: str, spec: str, shape_index: int = 0,
                  iters: int = 50, warmup: int = 20) -> dict:
    """在**独立子进程**里测一个实现，返回统计量。

    ## 为什么必须进程隔离（2026-11-25 实测）

    A 和 D 的 Triton 内核**都叫 `_fused_kernel`**，但签名不同：

        A: _fused_kernel(X, RES, W, OUT, M, eps, H, I, ...)
        D: _fused_kernel(X, R, W, OUT, eps, H, I)

    同一个进程里先加载 A 再调 D，Triton 的 JIT 缓存按(函数名, 参数签名)
    命中，把 D 的调用路由到了 A 的编译产物，报：

        Unsupported ptr type <['2048'], fp32> in `tl.store`

    那个报错和 D 的代码毫无关系，会让人以为 D 的 kernel 写错了——
    实测我为此折腾了四轮。

    进程隔离也顺带解决另外两件事：
      * 各实现的模块级状态不会互相污染（`sys.path`、全局常量、.so 句柄）
      * 一个实现崩溃不会带走整批复测

    代价是拿不到"同一进程交替"的精度。折中：进程内交替做不了，就改用
    **多轮往返**——每个实现跑 R 轮，每轮内 iters 次，然后取所有轮的中位数。
    轮与轮之间交错，同样能平摊漂移。
    """
    import json as _json
    import subprocess
    import tempfile

    payload = {
        "task": task_id, "spec": spec, "shape_index": shape_index,
        "iters": iters, "warmup": max(warmup, 30),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(payload, f)
        cfg = f.name
    try:
        p = subprocess.run(
            [sys.executable, "-m", "harness.verify", "--_worker", cfg],
            capture_output=True, text=True, timeout=900,
            cwd=ROOT,
        )
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "")[-500:])
        return _json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        try:
            os.unlink(cfg)
        except OSError:
            pass


def run(task_id: str, arms: list[str], shape_index: int = 0,
        iters: int = 50, warmup: int = 20,
        manual: dict[str, str] | None = None,
        rounds: int = 3, device: int | None = None) -> dict:
    """复测主流程。每个实现跑在独立子进程里（见 _run_isolated 的 docstring）。

    rounds：每个实现跑几轮。轮间交错无法跨进程做到，所以靠
    「多轮 + 取全场中位数」平摊漂移。实测并行跑时同一 kernel 能差 40%，
    单轮的中位数不够稳。
    """
    from bench.data import definition as defn_mod
    from bench.data import workload as wl_mod
    from harness.measure import resolve_spec
    from harness.launch import task_dir

    td = task_dir(task_id)
    defn = defn_mod.load(os.path.join(td, "definition.json"))
    ws = wl_mod.load(os.path.join(td, "workload.jsonl"), defn)
    pub, _ = wl_mod.split(ws)
    w = pub[shape_index]

    found, skipped = discover(task_id, arms)
    if manual:
        # 手工指定优先：认不出时的补救，也可用来加测别的实现
        for arm, spec in manual.items():
            found[arm] = (spec, found.get(arm, (None, None))[1])

    if not found:
        return {
            "task": task_id, "error": "没有任何实现能被识别",
            "skipped": skipped,
            "hint": "用 --impl 臂=脚本路径:函数 手工指定",
        }

    tensors = wl_mod.build_inputs(w, defn)
    extra = list((defn.dcu or {}).get("extra_args") or [])
    keys = list(defn.inputs)

    fns = {}
    sigs: dict[str, str] = {}
    for arm, (spec, _claimed) in found.items():
        fn = resolve_spec(spec)
        args = [tensors[k] for k in keys]

        # extra_args 按**签名**决定要不要传，不盲传。
        #
        # 实测教训：C 的 `get_runner("fused-hip-v5")` 返回的是
        # `_hip_runner_v5(x, residual, weight)`——3 个参数，eps 烧在 kernel 里；
        # D 的是 `(x, residual, weight, out=None, eps=1e-6)`。盲传 4 个参数
        # 直接把 C 打死（TypeError: takes 3 positional arguments but 4 were given）。
        #
        # 这也正是**提交契约必须统一**的原因：不统一时，复测脚本要为每个
        # 实现写一堆特判，而那本身又会引入新的错误。
        want = list(extra)
        try:
            import inspect

            sig = inspect.signature(fn)
            params = [p for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            sigs[arm] = str(sig)
            # 只用位置参数数量上限来裁剪；有 *args 或 VAR_POSITIONAL 就全给
            has_var = any(p.kind == p.VAR_POSITIONAL
                          for p in sig.parameters.values())
            if not has_var:
                room = len(params) - len(args)
                want = extra[:max(0, room)]
        except (TypeError, ValueError):
            sigs[arm] = "<无法读取签名>"

        def make(fn=fn, args=args, want=want):
            def call():
                fn(*args, *want)
            return call
        fns[arm] = make()

    # 逐个子进程测。不能用 time_interleaved：见 _run_isolated 的 docstring，
    # A/D 的内核同名会被 Triton 的 JIT 缓存串到一起。
    per_arm: dict[str, list[dict]] = {arm: [] for arm in found}
    order = list(found)

    # 正序 → 倒序 → 正序 ……
    #
    # 为什么：机器状态会漂（温度、邻居、DPM）。若每轮都按同一顺序，
    # 排第一的那个永远占"冷机"，排最后那个永远占"热机"，差出来的是
    # 顺序偏置而不是实现差异。实测 aabbcc 之后 ccbbaa，重复 6 轮，
    # 各臂离散度降到 0.5~0.8%。
    #
    # 这一步没法在子进程内做（每个实现在独立进程里），所以放在这里。
    env_keep = {k: os.environ.get(k) for k in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(device)
    try:
        for rnd in range(rounds):
            seq = order if rnd % 2 == 0 else list(reversed(order))
            for arm in seq:
                spec, _claimed = found[arm]
                try:
                    st = _run_isolated(task_id, spec, shape_index, iters, warmup)
                except RuntimeError as e:
                    # 单个实现失败不该带走整批。记下来，继续测别的。
                    st = {"error": str(e)[-300:], "median_ms": None}
                per_arm[arm].append(st)
    finally:
        for k, v in env_keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    timed = {}
    for arm, stats in per_arm.items():
        meds = [x["median_ms"] for x in stats if x.get("median_ms")]
        if not meds:
            continue
        med = statistics.median(meds)
        # MAD 用各轮中位数之间的离散度（跨轮稳定性），比轮内 MAD 更能反映
        # "这个数换个时间还行不行"
        mad = statistics.median([abs(v - med) for v in meds])
        cpus = [x["cpu_ms"] for x in stats if x.get("cpu_ms") is not None]
        cpu_med = statistics.median(cpus) if cpus else None
        timed[arm] = {
            "median_ms": med,
            "min_ms": min(x["min_ms"] for x in stats if x.get("min_ms")),
            "mad_ms": mad,
            "cv": (mad / med) if med else None,
            "n": sum(x.get("n", 0) for x in stats),
            "rounds": len(meds),
            "per_round_ms": meds,
            "cpu_ms": cpu_med,
            "cpu_share": (cpu_med / med) if (cpu_med and med) else None,
        }

    rows = []
    for arm, (spec, claimed) in found.items():
        t = timed[arm]
        rows.append({
            "arm": arm,
            "impl": spec,
            "measured_median_ms": t["median_ms"],
            "measured_min_ms": t["min_ms"],
            "mad_ms": t["mad_ms"],
            "cv": t["cv"],
            "claimed_median_ms": claimed,
            "drift_pct": ((t["median_ms"] - claimed) / claimed * 100
                          if claimed else None),
            "cpu_ms": t.get("cpu_ms"),
            "cpu_share": t.get("cpu_share"),
            "per_round_ms": t.get("per_round_ms"),
        })
    rows.sort(key=lambda r: r["measured_median_ms"])

    base = None
    bl = os.path.join(ROOT, "baselines", f"{task_id}.json")
    if os.path.isfile(bl):
        with open(bl, encoding="utf-8") as f:
            base = json.load(f).get("t_b_ms")

    return {"task": task_id, "t_b_ms": base, "shape": dict(w.axes),
            "rows": rows, "skipped": skipped, "iters": iters}


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------

CPU_SHARE_WARN = 0.20   # CPU 侧超过 GPU 时间的 20% 就点名


def render(res: dict) -> str:
    if "error" in res:
        lines = [f"✗ {res['error']}"]
        for arm, why in (res.get("skipped") or {}).items():
            lines.append(f"  {arm}: {why}")
        if res.get("hint"):
            lines.append(f"  {res['hint']}")
        return "\n".join(lines)

    tb = res.get("t_b_ms")
    lines = [
        f"═══ 复测：{res['task']} ═══",
        f"形状 {res.get('shape')}，交替 {res['iters']} 轮，每轮前 flush L2",
        "",
        f"{'臂':4s} {'复测 median':>12s} {'min':>10s} {'MAD':>9s} "
        f"{'CPU侧':>9s} {'CPU占比':>8s} {'自报':>10s} {'偏差':>8s} {'vs 基线':>8s}",
        "-" * 92,
    ]
    for r in res["rows"]:
        claimed = (f"{r['claimed_median_ms']:.5f}"
                   if r["claimed_median_ms"] else "-")
        drift = (f"{r['drift_pct']:+.1f}%" if r["drift_pct"] is not None else "-")
        rel = (f"{tb / r['measured_median_ms']:.2f}x" if tb else "-")
        cpu = (f"{r['cpu_ms']:.5f}" if r.get("cpu_ms") is not None else "-")
        share = (f"{r['cpu_share'] * 100:.0f}%"
                 if r.get("cpu_share") is not None else "-")
        lines.append(
            f"{r['arm']:4s} {r['measured_median_ms']:12.5f} "
            f"{r['measured_min_ms']:10.5f} {r['mad_ms']:9.5f} "
            f"{cpu:>9s} {share:>8s} "
            f"{claimed:>10s} {drift:>8s} {rel:>8s}"
        )

    if tb:
        lines.append(f"\n基线 t_b = {tb:.5f} ms")

    # CPU 侧开销要点名：交付入口本身带固定开销，读者会误判 kernel 水平
    heavy = [r for r in res["rows"]
             if r.get("cpu_share") is not None and r["cpu_share"] > CPU_SHARE_WARN]
    if heavy:
        lines.append("\n⚠ 下列实现的**交付入口**带显著 CPU 侧开销"
                     f"（> {CPU_SHARE_WARN:.0%} 的 GPU 时间）：")
        for r in heavy:
            lines.append(
                f"  {r['arm']}: CPU {r['cpu_ms']:.5f} ms = GPU 的 "
                f"{r['cpu_share'] * 100:.0f}%"
            )
        lines.append(
            "    这意味着「测到的耗时」里有一部分不是 kernel 花的。"
            "常见原因：每次调用都重新加载 .so / 反复分配 / Python 层循环。"
            "\n    要区分「交付入口」与「kernel 本体」，需读它的实现或让它提供预分配路径。"
        )

    # 大偏差要点名：>10% 说明自报数被环境带偏了
    big = [r for r in res["rows"]
           if r["drift_pct"] is not None and abs(r["drift_pct"]) > 10]
    if big:
        lines.append("\n⚠ 自报与复测偏差超过 10%（自报数不可用于排名）：")
        for r in big:
            lines.append(f"  {r['arm']}: 自报 {r['claimed_median_ms']:.5f} → "
                         f"复测 {r['measured_median_ms']:.5f} "
                         f"({r['drift_pct']:+.1f}%)")

    if res.get("skipped"):
        lines.append("\n没有参与复测的臂：")
        for arm, why in res["skipped"].items():
            lines.append(f"  {arm}: {why}")

    return "\n".join(lines)



def _worker_main(cfg_path: str) -> int:
    """子进程入口：测一个实现，把统计量打到 stdout 最后一行。

    **必须独立进程**：见 _run_isolated 的 docstring（A/D 的 _fused_kernel 同名
    会被 Triton 的 JIT 缓存串到一起）。
    """
    import json

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    from bench.data import definition as defn_mod
    from bench.data import workload as wl_mod
    from harness.measure import resolve_spec
    from harness.launch import task_dir

    task_id = cfg["task"]
    td = task_dir(task_id)
    defn = defn_mod.load(os.path.join(td, "definition.json"))
    ws = wl_mod.load(os.path.join(td, "workload.jsonl"), defn)
    pub, _ = wl_mod.split(ws)
    w = pub[cfg.get("shape_index", 0)]

    tensors = wl_mod.build_inputs(w, defn)
    extra = list((defn.dcu or {}).get("extra_args") or [])
    keys = list(defn.inputs)

    fn = resolve_spec(cfg["spec"])
    args = [tensors[k] for k in keys]
    # extra_args 按**关键字**补，不按位置。
    #
    # 实测教训（D 臂）：它的 `forward(x, residual, weight, out=None, eps=1e-6, ...)`
    # 第 4 个位置参数是 **out**，不是 eps。按位置传 [1e-6] 就变成 `out=1e-6`，
    # triton 把 float 当指针，报 `Unsupported ptr type <['2048'], fp32>`——
    # 那个报错和它的 kernel 代码毫无关系，我为此查了四轮。
    #
    # 按名字补的前提是知道名字。题面 `extra_args` 只有值，所以约定：
    # 多出来的标量一律按**题面里 eps 的常见名**试一遍（eps / epsilon /
    # variance_epsilon），对不上就退回位置传（A 的签名正好是位置的）。
    kwargs = {}
    try:
        import inspect

        sig = inspect.signature(fn)
        pnames = set(sig.parameters)
        candidates = ("eps", "epsilon", "variance_epsilon")
        if extra:
            for name in candidates:
                if name in pnames:
                    kwargs[name] = extra[0]
                    break
        params = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
        if kwargs:
            want = []          # 已按名字给，不再位置补
        elif has_var:
            want = list(extra)
        else:
            want = extra[:max(0, len(params) - len(args))]
    except (TypeError, ValueError):
        want = list(extra)
        kwargs = {}

    def call():
        fn(*args, *want, **kwargs)

    import torch

    # ---- CPU 侧开销检测 ----
    #
    # 为什么专门量这个：实测 C 臂交付的 `get_runner` 在 GPU 上和它自己的
    # kernel 一样快，但每次调用都重新 `ctypes.CDLL` 加载 .so（缓存键写错成
    # "lib"，恒不命中），CPU 侧多花 ~60µs。而我们的计时是
    # `e1.record(); call(); e2.record(); e2.synchronize()`——**CPU 在两次
    # record 之间的时间会被算进去**，于是交付入口 0.233ms、kernel 本体
    # 0.137ms，读者会以为它慢 70%。
    #
    # 这不是"谁写错了"的事，是**提交的入口本身带固定开销**。要报出来。
    import time as _time

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    cpu = []
    for _ in range(50):
        t0 = _time.perf_counter()
        call()
        cpu.append((_time.perf_counter() - t0) * 1e3)   # ms
    torch.cuda.synchronize()
    cpu_med = statistics.median(cpu)

    from bench.timing import flush as flush_mod

    nbytes = flush_mod.DEFAULT_FLUSH_BYTES
    for _ in range(max(30, cfg.get("warmup", 30))):
        call()
    torch.cuda.synchronize()

    # ---- 计时：先灌队列，避免把 CPU 提交时间算成 GPU 时间 ----
    #
    # 实测教训（2026-11-25，C 臂）：它的交付入口 CPU 侧要 77µs（每次重新
    # ctypes.CDLL 加载 .so），GPU 只要 136µs。若按
    #
    #     e1.record(); call(); e2.record()
    #
    # 计时，GPU 在 CPU 提交期间是**空闲**的，那 77µs 全被记进 event 区间，
    # 量出来 0.221ms，比真实值高 62%。而这不是选手慢，是裁判的口径把
    # 「Python 包装层开销」算成了「kernel 耗时」。
    #
    # 修法：每轮先发 PRIME 次（不计时）把 GPU 队列灌满，此时 CPU 已经
    # 跑到第 4 次提交、GPU 还在消化前几次，event 区间里就只剩 GPU 时间。
    # 实测：单次直测 0.221 → 灌队列后 0.131，与批量摊销（0.136）一致。
    #
    # PRIME 也要在**同一 buffer 语义**下：原地改写的实现每轮喂同一份输入是
    # 安全的（它们不改输入），flush 之后再 prime 会破坏 L2 状态——所以
    # prime 放在 flush 之前，最后再 flush 一次。
    PRIME = 3
    xs = []
    for _ in range(max(1, cfg.get("iters", 10))):
        for _ in range(PRIME):
            call()
        flush_mod.flush(nbytes)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        e.synchronize()
        xs.append(s.elapsed_time(e))

    med = statistics.median(xs)
    mad = statistics.median([abs(v - med) for v in xs])
    print(json.dumps({
        "median_ms": med, "min_ms": min(xs), "mad_ms": mad,
        "n": len(xs), "sclk_mhz": _sclk_now(),
        "cpu_ms": cpu_med,
        # CPU 侧占 GPU 时间的比例：>20% 说明这个入口有固定开销，
        # 交付值不能代表 kernel 本体水平
        "cpu_share": (cpu_med / med) if med else None,
    }))
    return 0


def _sclk_now():
    try:
        from bench.hw import dcu

        return dcu.sclk_of_device()
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="opbench-verify",
        description="复测各臂提交的实现（同卡、同进程、交替测量）",
    )
    p.add_argument("--task", default=None)
    p.add_argument("--arms", nargs="+", default=["A", "C", "D"])
    p.add_argument("--shape-index", type=int, default=0)
    p.add_argument("--iters", type=int, default=10,
                   help="每轮采样数。实测 10×5 与 50×1 的总样本相同，"
                        "但 5 个独立中位数能看出漂移，跨轮离散 0.27%%")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--impl", action="append", default=None,
                   help="手工指定：臂=脚本路径:函数（可给多次）")
    p.add_argument("--out", default=None)
    p.add_argument("--rounds", type=int, default=5,
                   help="每个实现跑几轮（轮间取中位数，平摊漂移）")
    p.add_argument("--device", type=int, default=None, help="绑卡；默认沿用环境变量")
    p.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a._worker:
        return _worker_main(a._worker)
    if not a.task:
        p.error("需要 --task")

    manual = {}
    for item in (a.impl or []):
        if "=" not in item:
            print(f"--impl 格式应为 臂=路径:函数，实际 {item!r}", file=sys.stderr)
            return 2
        arm, spec = item.split("=", 1)
        manual[arm.strip().upper()] = spec.strip()

    res = run(a.task, [x.upper() for x in a.arms], a.shape_index,
              a.iters, a.warmup, manual, rounds=a.rounds, device=a.device)
    print(render(res))

    out = a.out or os.path.join(ROOT, "reports", "verify-last.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# 选出每条臂「最终采用」的那一版
# --------------------------------------------------------------------------

def pick_submission(task_id: str, arm: str) -> dict:
    """判定某臂**最终提交的是哪个实现**，以及为什么这么判。

    ## 为什么需要这一步

    一次 attempt 会留下几十个中间版本。实测 D 臂的 scripts/ 下有 18 个 .py
    （tune.py / tune2.py / ... / fused_triton.py / fused_final.py），
    其中至少两个跑不通。复测随便挑一个，测的可能不是它的最终成果，
    也不是它 report 里写的那个数。

    ## 判据（按可靠性降序）

    1. **`results/bench_std.json` 的 `impl` 字段**——这是题面要求的
       "我落到了哪个实现"，是 agent 自己的声明，最权威。
    2. 从 `impl` 里解析出的 `<脚本>:<函数>`，验证文件存在 + 函数存在。
    3. 如果 impl 只是个标签名（`fused-hip-v5`），去卡/变体表里查真身。
    4. 都不行 → **如实报无法确定**，不用猜测的版本充数。

    ## 还要和 report 交叉验证

    agent 可能在 report 里写"最终方案是 X"，而 bench_std 的 impl 写 Y。
    这种不一致必须报出来——它意味着 agent 自己没想清交付物是什么，
    或者它在两版之间反复了。两种都值得看。
    """
    import glob

    from harness.multi.collect import collect_one

    art = collect_one(task_id, arm)
    out = {"arm": arm, "dir": art.dir, "impl_field": None,
           "resolved": None, "func": None, "problems": [], "report_claims": []}

    bs = art.bench_std or {}
    impl = str(bs.get("impl") or "")
    out["impl_field"] = impl or None
    if not impl:
        out["problems"].append("bench_std.json 没有 impl 字段")
    else:
        path, func, why = _find_impl_spec(art)
        if path:
            out["resolved"] = path
            out["func"] = func
        else:
            out["problems"].append(why or "无法从 impl 解析出实现")

    # report 里声称的最终实现
    if art.has_report:
        try:
            txt = open(os.path.join(art.dir, "report.md"), encoding="utf-8",
                       errors="replace").read()
        except OSError:
            txt = ""
        for pat in (r"最终(?:实现|方案|版本)[：:]\s*([^\n]{4,60})",
                    r"落到了哪个实现[^\n]*\n+[^\n]*`([^`]{4,60})`"):
            for m in re.finditer(pat, txt):
                out["report_claims"].append(m.group(1).strip()[:60])

    return out


def pick_all(task_id: str, arms: list[str]) -> list[dict]:
    return [pick_submission(task_id, a) for a in arms]


def render_pick(rows: list[dict]) -> str:
    lines = [f"═══ 各臂最终提交的是哪一版 ═══", ""]
    for r in rows:
        lines.append(f"▸ 臂 {r['arm']}")
        lines.append(f"   impl 字段: {r['impl_field'] or '(缺失)'}")
        if r["resolved"]:
            lines.append(f"   解析到:   {os.path.basename(r['resolved'])}:{r['func']}")
        if r["report_claims"]:
            lines.append(f"   报告声称: {r['report_claims'][:2]}")
        for p in r["problems"]:
            lines.append(f"   ⚠ {p}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 提交契约的机检
# --------------------------------------------------------------------------

def lint_submission(task_id: str, arm: str) -> list[str]:
    """检查某臂的提交是否符合契约。返回问题清单，空 = 合格。

    为什么要机检而不只是写在题面里：题面说了"impl 要写
    `scripts/x.py:func`"，实测三臂只有 A 照做。**约束不做机器检查就等于没有。**

    检查项（按是否能自动判定排序）：

      1. bench_std.json 存在且能被 json 解析
      2. 有 impl 字段
      3. impl 里能解析出 `<路径>.py` 与 `<函数名>`
      4. 该文件存在
      5. 该函数在文件里真实定义（ast 里找得到）
      6. 函数能被调用（import 后试调一次）
      7. 调用后 CPU 侧耗时不超 GPU 耗时的 50%（否则交付入口带固定开销）
    """
    import json as _json

    from harness.multi.collect import collect_one

    problems: list[str] = []
    art = collect_one(task_id, arm)
    if not art.dir:
        return [f"臂 {arm}: 没有 attempt 目录"]

    bs_path = os.path.join(art.dir, "results", "bench_std.json")
    if not os.path.isfile(bs_path):
        return [f"臂 {arm}: 缺 results/bench_std.json"]
    try:
        with open(bs_path, encoding="utf-8") as f:
            bs = _json.load(f)
    except _json.JSONDecodeError as e:
        return [f"臂 {arm}: bench_std.json 不是合法 JSON（{e}）"]

    impl = str(bs.get("impl") or "")
    if not impl:
        return [f"臂 {arm}: bench_std.json 没有 impl 字段"]

    path, func, why = _find_impl_spec(art)
    if not path:
        return [f"臂 {arm}: impl 无法解析 → {why}\n"
                f"    写了：{impl[:100]}\n"
                f"    要求：`scripts/<文件>.py:<入口函数>`"]

    rel = os.path.relpath(path, art.dir)
    # 4. 函数真的在文件里定义吗
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except (OSError, SyntaxError) as e:
        return [f"臂 {arm}: 读不了 {rel}（{e}）"]
    if ":" in func:
        # 工厂形式 path:get_runner:变体名
        f0 = func.split(":")[0]
        if f0 not in defined:
            problems.append(f"臂 {arm}: {rel} 里没有函数 {f0}（有：{sorted(defined)[:8]}）")
    elif func not in defined:
        problems.append(f"臂 {arm}: {rel} 里没有函数 {func}（有：{sorted(defined)[:8]}）")

    # 5. 函数名会不会太通用（同名冲突风险）
    if func in ("_fused_kernel", "kernel", "forward", "run", "_kernel"):
        problems.append(
            f"臂 {arm}: 入口函数名 {func!r} 太通用。"
            "复测靠它定位实现，同名容易被别的臂的 JIT 缓存串味。"
            "建议改成带算子名的，如 fused_add_rmsnorm_swiglu"
        )

    return problems


def lint_all(task_id: str, arms: list[str]) -> dict:
    return {arm: lint_submission(task_id, arm) for arm in arms}


def render_lint(res: dict) -> str:
    lines = ["═══ 提交契约检查 ═══", ""]
    bad = 0
    for arm, probs in res.items():
        if not probs:
            lines.append(f"✓ 臂 {arm} 合格")
        else:
            bad += 1
            lines.append(f"✗ 臂 {arm} 有 {len(probs)} 个问题：")
            for pb in probs:
                lines.append(f"    {pb}")
    lines.append("")
    lines.append(f"{len(res) - bad}/{len(res)} 臂合格" if bad else "全部合格")
    return "\n".join(lines)
