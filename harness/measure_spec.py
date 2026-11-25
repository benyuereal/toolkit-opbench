"""测量口径：从 baselines/<task>.json 固化，注入到每个臂的 prompt。

为什么必须在这里做：
    题面只给了张量语义（比如 q 是 [S,H,576]），没给 H/SKV/TOPK/S 的具体数值。
    实测 A 臂自选 SKV=16384/TOPK=2048，而 C/D 用 SKV=8192/TOPK=256，
    S=512 时耗时 2542 us vs 163 us —— 差 15 倍。
    那不是"优化差异"，是**测的不是同一个算子**。

所以口径由 harness 统一下发：三臂逐字相同，差异只剩"有没有那个 skill 环境"。
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _std_schema(shapes: list, peak_id: str, peakv) -> str:
    """按本题实际 shapes 生成 JSON 示例。

    不能写死一个例子：实测把 mla 的 `SKV/TOPK/S` 当示例注入给 norm 题，
    agent 会照着示例的 axes 写，和本题口径对不上。
    """
    ex = []
    for s in shapes[:1]:
        axes = s.get("axes") or {}
        ex.append(
            '    {"uuid": "%s", "axes": %s,\n'
            '     "median_ms": 0.00000, "min_ms": 0.00000, '
            '"iters": 50, "warmup": 10,\n'
            '     "l2_flush_bytes": 33554432}'
            % (s.get("uuid", "?"),
               json.dumps(axes, ensure_ascii=False, separators=(", ", ": ")))
        )
    den = (f'"denominator": {{"peak_id": "{peak_id}", "value_gbs": {peakv}}},'
           if peak_id else "")
    return ("```json\n{\n  \"task\": \"<task id>\",\n  \"shapes\": [\n"
            + ",\n".join(ex) + "\n  ],\n  "
            + (den + "\n  " if den else "")
            + "\"impl\": \"scripts/<文件>.py:<入口函数>  —  <一句话说明>\"\n}\n```\n")


def baseline_path(task_id: str) -> str:
    return os.path.join(ROOT, "baselines", f"{task_id}.json")


def _shapes_from_workload(task_id: str) -> list[dict] | None:
    """从题目自己的 workload.jsonl 取 public 档位。

    ## 为什么口径档位要以 workload 为准，不是基线

    实测踩过：给三道题加了 `pub-decode-128` 档，但基线还没重跑，
    于是题面里的档位表**完全没有 decode 档**——agent 根本不知道要测它。
    而 workload 才是"这道题考哪些形状"的权威定义，基线只是"其中某些
    形状测过没有"。

    返回值形状与基线里的 shapes 一致（`[{"uuid":..., "axes":{...}}]`），
    这样下游不用改。取不到就返回 None，让调用方回退到基线。
    """
    try:
        from bench.data import definition as defn_mod
        from bench.data import workload as wl_mod
        from harness.run import task_dir

        td = task_dir(task_id)
        d = defn_mod.load(os.path.join(td, "definition.json"))
        ws = wl_mod.load(os.path.join(td, "workload.jsonl"), d)
        pub, _hold = wl_mod.split(ws)
    except Exception:
        return None
    if not pub:
        return None
    out = []
    for w in pub:
        axes = dict(d.const_axes())
        axes.update(w.axes)
        out.append({"uuid": w.uuid, "axes": axes})
    return out


def render(task_id: str) -> str:
    """返回要追加到 prompt 末尾的测量口径段落。

    档位表来自 **workload.jsonl 的 public 段**（题目定义），
    基线只提供 t_b / t_sol / peak。这样新加的档位立刻出现在题面里，
    不必等基线重跑。
    """
    p = baseline_path(task_id)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
    except Exception:
        return ""
    shapes = _shapes_from_workload(task_id) or (b.get("shapes") or [])
    if not shapes:
        return ""

    out = ["\n\n---\n\n## 测量口径（评分基准，不可更改）\n"]
    out.append(
        "\n以下档位就是评分的基准。**必须每个档位都测到，且 axes 逐项一致**。\n"
        "自己改 axes（放大 SKV/TOPK、换 M、换 H）测出来的数**不进对比**，"
        "因为那和基线不是同一个算子。\n\n"
    )
    out.append("| uuid | axes |\n|---|---|\n")
    for s in shapes:
        axes = s.get("axes") or {}
        kv = ", ".join(f"{k}={v}" for k, v in axes.items())
        out.append(f"| `{s.get('uuid', '?')}` | {kv} |\n")

    tb = b.get("t_b_ms")
    tsol = b.get("t_sol_ms")
    if tb is not None:
        out.append(f"\n基线 `t_b = {tb:.5f} ms`")
    if tsol is not None:
        out.append(f"，理论下界 `t_sol = {tsol:.5f} ms`"
                   f"（上限 {tb/tsol:.2f}x）" if tb else "")
    out.append("。\n")

    peak = b.get("peak_id")
    peakv = b.get("peak_value")
    if peak:
        out.append(f"\n达成率分母：`{peak}`")
        if peakv is not None:
            out.append(f" = {peakv}")
        out.append("。用别的分母算出来的百分比不接受。\n")

    out.append(
        "\n### 结果必须写成 `results/bench_std.json`\n\n"
        "这是 harness 唯一会读的性能文件。自己的 log/json 可以另外留，"
        "但**没有 bench_std.json 就等于没测**。格式：\n\n"
    )
    out.append(_std_schema(shapes, peak, peakv))
    out.append(
        "\n要求：\n"
        "- `axes` 必须与上表**逐项相同**（多一个键就按基准那组裁掉，少一个键直接判无效）。\n"
        "- `median_ms` 用 GPU event 计时，前 `warmup` 轮丢弃，每轮前 flush L2。\n"
        "- 同时记录 `min_ms`：median 受调度抖动影响，min 更接近真实可达。\n"
    )
    out.append(_impl_contract())
    return "".join(out)


def _impl_contract() -> str:
    """交付契约：`impl` 字段必须能让第三方**不看你报告就复现**。

    ## 为什么必须写死格式

    实测三臂各自交了 `impl`，没有两个写法一样：

        A: "triton 3.6.0 :: scripts/fused_add_rmsnorm_swiglu.py::fused_add_rmsnorm_swiglu"
        C: "fused-hip-v5 = 自研 HIP kernel kernels/norm_act_swiglu_v5.hip (hipcc, ...)"
        D: "triton.jit:_fused_kernel (custom fused Triton kernel, ...)"

    其中 D 写的 `_fused_kernel` 是个**在它的文件里不存在的名字**（真身叫
    `fused_forward`，在 `fused_final.py`），而它 scripts/ 下同时躺着 18 个
    .py（tune.py / tune2.py / ... / fused_triton.py / fused_final.py），
    复测方只能靠猜——第一次猜到了编译不过的旧版本。

    这不是 agent 的错，是**我们没给契约**。所以现在写死。
    """
    return (
        "\n### `impl` 字段的格式（复测要用，必须照写）\n\n"
        "评审会把各臂的实现收上来**重新测一遍**（同卡、同进程、交替测量），\n"
        "因为并行跑时 HBM 是共享的、自报数字会被邻居带偏。要吃上这步，\n"
        "`impl` 必须能让评审**不看你的报告**就知道跑哪一行代码。\n\n"
        "格式（严格）：\n\n"
        "```\n"
        "<脚本相对路径>:<入口函数名>  —  <一句话说明>\n"
        "```\n\n"
        "例：\n\n"
        "```\n"
        "scripts/fused_add_rmsnorm_swiglu.py:fused_add_rmsnorm_swiglu  —  Triton 单 kernel，三段全融合\n"
        "```\n\n"
        "硬性要求：\n\n"
        "1. **`:` 前必须是真实存在的 .py 文件路径**（相对当前目录），"
        "`:` 后必须是该文件里真实定义的函数名。\n"
        "2. **签名必须是 `fn(<题面 inputs 的键序>, eps=<默认值>)`**。\n"
        "   即：张量按 `inputs` 的顺序排在最前面，**标量一律用关键字/默认值**，"
        "不要插在张量中间。\n"
        "   反例（实测踩过）：`fn(x, residual, weight, out=None, eps=1e-6)`——"
        "第 4 个位置参数是 `out` 而不是 `eps`，复测按位置传 `1e-6` 就变成 "
        "`out=1e-6`，kernel 报 `Unsupported ptr type ... fp32`，"
        "而那个报错和你的代码毫无关系。\n"
        "3. 该函数**不要要求额外的准备工作**（预编译、先 build .so、手动传 buffer）。"
        "如果要，就在同目录放一个 `build.sh` 并在报告里写清，评审会先跑它。\n"
        "4. 同一个 scripts/ 里可以有很多调试脚本，但**只有 impl 指的那一个会被复测**。"
        "别把最终实现藏在 `tune5.py` 里而 `impl` 写成 `fused-hip-v5`——"
        "那个名字复测方解析不了，你的成绩会算不出来。\n"
        "5. **内核函数名不要用 `_fused_kernel` 这类通用名**，或者至少保证"
        "「入口函数」是唯一的。实测两个臂的内核都叫 `_fused_kernel` 但签名不同，"
        "复测方在同一个进程里先后加载会被 Triton 的 JIT 缓存串到一起——"
        "现在复测已改成逐个子进程，但换个工具就可能再撞。\n"
        "6. 如果你交付的是编译出来的 .so / .hip，`impl` 仍要指到 Python 入口：\n"
        "   `scripts/run_v5.py:run  —  HIP kernel kernels/xxx.hip（build.sh 编）`\n"
        "7. **入口函数里不要放会重复执行的慢操作**。判定会额外量一次\n"
        "   「CPU 侧耗时」——即调用你的函数所花的 Python 时间。若它超过\n"
        "   GPU 耗时的 20%，会单独标出来，因为那说明交付入口本身有固定开销，\n"
        "   你报的数不能代表 kernel 水平。\n"
        "   实测踩过的例子：`if \"lib\" not in _LIB or ...` 里缓存键写错，\n"
        "   导致每次调用都重新 `ctypes.CDLL` 加载 .so，CPU 侧多 60µs，\n"
        "   交付入口 0.233ms vs kernel 本体 0.137ms（差 70%）。\n"
        "   自己用 `time.perf_counter()` 围住调用量一下就能发现。\n"
        "\n"
        "**如果 impl 解析不出来，本轮成绩按「无法复现」处理，不参与排名。**\n"
    )
