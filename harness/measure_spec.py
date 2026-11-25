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
            + "\"impl\": \"库.函数名 + 版本\"\n}\n```\n")


def baseline_path(task_id: str) -> str:
    return os.path.join(ROOT, "baselines", f"{task_id}.json")


def render(task_id: str) -> str:
    """返回要追加到 prompt 末尾的测量口径段落。没有 baseline 时返回空串。"""
    p = baseline_path(task_id)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
    except Exception:
        return ""
    shapes = b.get("shapes") or []
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
    return "".join(out)
