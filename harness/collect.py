"""汇总：runs/ → reports/。

产出三臂对比表与 Δ：
  Δ_eff    有效结论率差
  Δ_halluc 幻觉率差（期望 <= 0）
  Δ_calib  校准误差
胜负线跑前定死，这里只做计算与渲染。

## "只认产物"在这里怎么落地

不读 agent 的自述，只读：
  * report.md —— 唯一允许的结论载体（散文）
  * solution.json —— claims 的机器可读形态（有则读）
  * results/*.json —— 数字的唯一出处
  * meta.json —— 这一轮的运行记录（tokens / 状态 / 哈希）

评分等级（judgment 类）直接复用 bench/score/judgment.py，保证与
`make score` 同一套口径。**这里不重新定义分数**，否则会出现
"报告说 100、score 说 60"的分裂。
"""
from __future__ import annotations

import json
import os
import re

ARMS = ("A", "B", "C", "D")
ATTEMPT = "attempt-1"


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _arm_dir(runs_dir: str, arm: str) -> str:
    return os.path.join(runs_dir, f"arm-{arm}")


# 隔离臂（有独立用户的，目前只有 A）的 cwd 在仓库外的 /tmp/opbench-run。
# 汇总必须同时扫这两处，否则 A 臂的结果会静默消失——那会让 Δ 变成
# 「C 减一个不存在的 A」，看起来像 C 大幅领先。
ISOLATED_ROOT = "/tmp/opbench-run"


def _candidate_roots(runs_dir: str, isolated_root: str | None = None) -> list[str]:
    roots = [runs_dir]
    iso = isolated_root
    if iso is None:
        iso = os.environ.get("OPBENCH_ISOLATED_ROOT", ISOLATED_ROOT)
    if iso and os.path.isdir(iso) and os.path.realpath(iso) != os.path.realpath(runs_dir):
        roots.append(iso)
    return roots


def _attempt_dirs(runs_dir: str, arm: str,
                  isolated_root: str | None = None) -> list[tuple[str, str]]:
    """[(task_id, attempt_dir)]，每个 task 一条（只认 attempt-1）。

    仓库内与隔离目录都扫，同一题以隔离目录为准（那是真正跑出来的那份，
    仓库内可能只是早先的预演）。
    """
    found: dict[str, str] = {}
    for root in _candidate_roots(runs_dir, isolated_root):
        base = os.path.join(root, f"arm-{arm}")
        if not os.path.isdir(base):
            continue
        for task_id in sorted(os.listdir(base)):
            d = os.path.join(base, task_id, ATTEMPT)
            if os.path.isdir(d):
                found[task_id] = d
    return sorted(found.items())


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _task_dir_of(task_id: str, root: str) -> str | None:
    """按 id 或目录名找题目录（目录名与 definition.id 可能不同）。"""
    direct = os.path.join(root, "tasks", task_id)
    if os.path.isfile(os.path.join(direct, "definition.json")):
        return direct
    for name in sorted(os.listdir(os.path.join(root, "tasks"))):
        p = os.path.join(root, "tasks", name, "definition.json")
        if name.startswith("_") or not os.path.isfile(p):
            continue
        d = _load_json(p)
        if d and d.get("id") == task_id:
            return os.path.join(root, "tasks", name)
    return None


def _definition_for(task_id: str) -> dict | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    direct = os.path.join(root, "tasks", task_id, "definition.json")
    if os.path.isfile(direct):
        return _load_json(direct)
    for name in sorted(os.listdir(os.path.join(root, "tasks"))):
        p = os.path.join(root, "tasks", name, "definition.json")
        if name.startswith("_") or not os.path.isfile(p):
            continue
        d = _load_json(p)
        if d and d.get("id") == task_id:
            return d
    return None


# ---------------------------------------------------------------- 各项判定

def has_effective_conclusion(report: str, artifacts: list[str],
                             kind: str = "kernel") -> tuple[bool, str]:
    """“有效结论”= 有可复跑数字 + 复跑方法。

    判法刻意保守：要么给出命令（可复跑方法），要么给出落盘的原始输出 +
    数字。两者都没有的一律不算——我们被“写了一堆但没有数”坑过。

    judgment 类（L3/L5）另按一套：正确答案是“通不通”，本来就没有性能数字。
    对它要求“有数字”会把每道 L5 都判成无效，Δ_eff 恒为 0——那测的是
    题型的口味，不是能力。所以判定题的判据是“有结论 + 有可复跑证据”。
    """
    if not report.strip():
        return False, "没有 report.md"
    if kind == "judgment":
        has_cmd = bool(re.search(r"(python3?\s|bash\s|sh\s+-c|命令[:：]|复跑)", report))
        has_raw = any(p.startswith("results/") for p in artifacts)
        if has_cmd and has_raw:
            return True, "有结论 + 复跑命令 + 原始输出"
        if has_cmd:
            return True, "有结论 + 复跑命令"
        return False, "没有复跑方法（命令或原始输出）"
    has_cmd = bool(re.search(r"(python3?\s|bash\s|sh\s+-c|命令[:：])", report))
    # 数字必须自成一体：左边界不能是字母数字或点（否则 python3 x.py 里的
    # "3 x" 会被当成 3 倍加速），右边界不能紧跟字母（排除 16MB 里的 16）。
    has_num = bool(
        re.search(r"(?<![\w.])-?\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|s|TFLOPS|TF|GB/s|%|x|×)(?![\w.])",
                  report, re.I)
    )
    has_raw = any(p.startswith("results/") for p in artifacts)
    if has_cmd and (has_num or has_raw):
        return True, "有命令 + 数字/原始输出"
    if has_num and has_raw:
        return True, "有数字 + 原始输出"
    if not has_num:
        return False, "报告里没有可核对的数字"
    return False, "有数字但没有复跑方法（命令或原始输出）"


def judged_score(task_id: str, attempt_dir: str) -> tuple[int, str] | None:
    """judgment 类：复用 bench/score/judgment。"""
    defn = _definition_for(task_id)
    if not defn or defn.get("kind") != "judgment":
        return None
    from bench.score import judgment

    report = _read(os.path.join(attempt_dir, "report.md"))
    if not report.strip():
        return 0, "没有 report.md"
    evidence = _evidence_from(attempt_dir, report)
    return judgment.score(report, evidence, defn)


# 证据片段的识别：只认"可核对的具体东西"，不认散文行。
# 之前把 report 的每一行都当证据，A 臂的报告被数成 216 条——那种计数
# 只会让"证据充分"变成"话说得多"。
_EVIDENCE_PATTERNS = (
    re.compile(r"^\s*```"),                      # 代码块
    re.compile(r"(RuntimeError|ValueError|TypeError|Error:|Traceback|报错原文)"),
    re.compile(r"(python3?\s+\S+\.py|bash\s+\S+\.sh|\$\s+\S+)"),  # 可复跑命令
    re.compile(r"(results/\S+|scripts/\S+)"),    # 产物路径
    # 数字在前、单位在后（"1.23 ms"），或指标名在前（"rel_l2 = 0.001"）
    re.compile(
        r"(?<![\w.])\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|s|TFLOPS|GB/s|%|x|×)(?![\w.])"
        r"|\b(rel_l2|max_abs|cos|ratio)\b\s*[=:≈]?\s*\d",
        re.I,
    ),
)


def _evidence_from(attempt_dir: str, report: str) -> list[str]:
    """证据 = 可核对的具体片段 + results/ 下的原始输出。

    只收三类，各自对应一种"能核"的东西：
      1. 报告里的代码块 / 报错原文 / 复跑命令 / 产物路径 / 带单位的数字
      2. results/ 下每个文件本身（含其前 4000 字符）
    散文行不算证据。一个 agent 写 200 行说明而没跑过任何命令，
    和写 5 行但贴了报错 + 命令，后者才叫有证据。
    """
    evidence: list[str] = []
    in_code = False
    for line in report.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            if in_code:
                evidence.append("```")   # 一个代码块算一条
            continue
        if in_code:
            continue  # 代码块内部已整体计过一次，不再逐行累加
        if any(p.search(line) for p in _EVIDENCE_PATTERNS):
            evidence.append(line.strip())

    rdir = os.path.join(attempt_dir, "results")
    if os.path.isdir(rdir):
        for name in sorted(os.listdir(rdir)):
            p = os.path.join(rdir, name)
            if os.path.isfile(p):
                evidence.append(os.path.join("results", name))
                evidence.append(_read(p)[:4000])

    # 原文本身也算一条：`evidence_required` 问的是"报告里有没有出现这类
    # 证据"，不是"有没有出现在我筛过的摘要里"。过滤只用于**计数**，
    # 不能反过来把证据种类判丢——否则写了 `import aiter` 的报告会被判
    # 成没有 import 证据。
    evidence.append(report)
    return evidence


def hallucination_rate(task_id: str, attempt_dir: str) -> tuple[float, dict]:
    """幻觉率 = 编造的声明数 / 总声明数。声明本身从产物里抽。"""
    from bench.check import hallucination
    from bench.data import solution as sol_mod

    claims = sol_mod.extract_claims(attempt_dir)
    registry = "/workspace/toolkit_skills/data/ops/libraries-registry.yaml"
    result = hallucination.check_solution(None, attempt_dir, registry)
    del claims  # check_solution 内部会抽，避免两套抽取逻辑不一致
    return result.get("rate", 0.0), result


def leak_hits(attempt_dir: str) -> dict[str, list[str]]:
    from bench.check import leak

    return leak.scan_dir(attempt_dir)


def _extract_t_ms(attempt_dir: str) -> dict | None:
    """从产物里抽被测实现的耗时，**按形状分开**。

    返回 {shape_key: ms}，或 None。

    为什么不能只返回一个数：不同 M 的耗时天差地别（rmsnorm M=16 是 38µs、
    M=8192 是 203µs）。取 min 会拿最小形状跟别的臂的大形状比，比出个假排名。
    这里按同名 axes（M/H 等）分组，跨臂比时只比**同名形状**。

    只认结构化 results/*.json，不读散文报告——bench 的数必须可复核。
    """
    import glob

    out: dict[str, float] = {}
    for p in glob.glob(os.path.join(attempt_dir, "results", "*.json")):
        try:
            obj = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item, shape in _walk_dicts(obj):
            t = item.get("timing")
            ms = None
            # 秒单位：B 臂用 median_s（实测踩过）
            for k in ("median_s", "t_s"):
                if isinstance(item.get(k), (int, float)) and not isinstance(item.get(k), bool):
                    ms = float(item[k]) * 1e3
                    break
            if ms is None and isinstance(t, dict) and isinstance(t.get("median_ms"), (int, float)):
                ms = float(t["median_ms"])
            if ms is None:
                # 兜底：认 "xxx_ms"，但**排除基准/对照类键**。
                # 实测踩过：B 臂的 results 里有 empty_loop_ms=0.00019（空循环
                # 基准）和 launch_floor_ms，把它们当 gating 耗时会让它"快 228 倍"。
                # 这些数是参照物，不是被测结果。
                SKIP = ("floor", "empty", "baseline", "ref", "torch", "unfused",
                        "loop", "overhead")
                cands = [v for k, v in item.items()
                         if k.endswith("_ms") and isinstance(v, (int, float))
                         and not isinstance(v, bool) and v > 0
                         and not any(sk in k.lower() for sk in SKIP)]
                if cands:
                    ms = float(min(cands))
            if ms is None:
                continue
            key = ",".join(f"{k}={v}" for k, v in sorted(shape.items())) or "default"
            out.setdefault(key, ms)
    return out or None


def _shape_of(item: dict) -> dict:
    """从 item 或它的嵌套容器里找形状字段。"""
    out = {}
    ax = item.get("axes")
    if isinstance(ax, dict):
        out.update({k: v for k, v in ax.items() if isinstance(v, int)})
    for k in ("M", "H", "E", "TOPK", "N", "K"):
        if k not in out and isinstance(item.get(k), int):
            out[k] = item[k]
    return out


def _walk_dicts(obj, inherited: dict | None = None):
    """递归产出 (dict, 继承的形状)。子节点的形状从父级继承并覆盖。"""
    if isinstance(obj, dict):
        sh = dict(inherited or {})
        sh.update(_shape_of(obj))
        yield obj, sh
        for v in obj.values():
            yield from _walk_dicts(v, sh)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v, inherited)


def collect(runs_dir: str, isolated_root: str | None = None,
            only_tasks: list[str] | None = None) -> dict:
    """扫 runs/，返回 {tasks: [...], arms: [...], table: {(task,arm): row}, deltas}

    `isolated_root` 传给 `_candidate_roots`；测试传空串可只扫给定目录，
    避免被本机 /tmp/opbench-run 的真实运行干扰。

    `only_tasks` 限定要汇报的题集。**必须传**：runs/ 里累积了历史评测的
    单元（gemm 的 timeout、l5、judge 的失败），全扫进报告会污染 Δ——
    之前 A 臂被算成"6 题"就是因为它目录里有历史残留。报告应该只反映
    "这次评测的题"。
    """
    # 先列出所有出现过任务的目录名（各臂可能只跑了一部分）
    all_tasks: list[str] = []
    for arm in ARMS:
        for task_id, _ in _attempt_dirs(runs_dir, arm, isolated_root):
            if task_id not in all_tasks:
                all_tasks.append(task_id)
    if only_tasks:
        keep = set(only_tasks)
        all_tasks = [t for t in all_tasks if t in keep]
    tasks = sorted(all_tasks)

    table: dict[tuple[str, str], dict] = {}
    keep = set(only_tasks) if only_tasks else None
    for arm in ARMS:
        for task_id, d in _attempt_dirs(runs_dir, arm, isolated_root):
            if keep is not None and task_id not in keep:
                continue
            meta = _load_json(os.path.join(d, "meta.json")) or {}
            report = _read(os.path.join(d, "report.md"))
            artifacts = meta.get("artifacts") or []
            for name in ("report.md", "solution.json"):
                if os.path.isfile(os.path.join(d, name)) and name not in artifacts:
                    artifacts.append(name)

            defn = _definition_for(task_id) or {}
            eff, eff_why = has_effective_conclusion(
                report, artifacts, kind=defn.get("kind", "kernel")
            )
            js = judged_score(task_id, d)
            hrate, hdetail = hallucination_rate(task_id, d)
            leaks = leak_hits(d)

            row = {
                "task": task_id,
                "arm": arm,
                "status": meta.get("status"),
                "wall_time_s": meta.get("wall_time_s"),
                "tokens": (meta.get("tokens") or {}).get("output_tokens"),
                "artifacts": len(artifacts),
                "has_report": bool(report.strip()),
                "effective": eff,
                "effective_why": eff_why,
                "judged": js[0] if js else None,
                "judged_why": js[1] if js else None,
                "halluc_rate": hrate,
                "halluc": hdetail,
                "leaks": leaks,
                "leaked": bool(leaks),
                # 性能：从 results 抽耗时，稍后跨臂算相对/roofline 达成率
                "t_ms": _extract_t_ms(d),
            }
            table[(task_id, arm)] = row

    # ---- 性能评分：跨臂算相对达成率 + roofline 达成率 ----
    perf_rows = _score_perf(table, tasks)

    deltas = _deltas(table, tasks)
    return {"tasks": tasks, "arms": list(ARMS), "table": table,
            "deltas": deltas, "perf": perf_rows}


def _score_perf(table: dict, tasks: list[str]) -> dict:
    """每题：用三臂里最快的当分母算相对达成率，并算 roofline 达成率。

    业主口径：性能占 80%，正确性硬门槛。这里只算性能部分，
    正确性由 bench.check.correctness 单独判（不通过的题不进排名）。
    """
    from bench.data import definition as defn_mod
    from bench.score import perf as perf_mod

    out = {}
    for task in tasks:
        # 跨臂比"同名形状"：先找出三臂都有的形状键，只在那些键上比。
        per_arm = {arm: table[(task, arm)].get("t_ms")
                   for arm in ARMS
                   if (task, arm) in table and not table[(task, arm)].get("leaked")}
        per_arm = {a: v for a, v in per_arm.items() if isinstance(v, dict) and v}
        if not per_arm:
            continue
        common = set.intersection(*[set(v) for v in per_arm.values()])
        # 没有完全共同的形状时，退化为"各自最大形状"（大形状最能代表稳态）
        if common:
            shape_key = sorted(common, key=lambda k: len(k))[0] if len(common) == 1 else \
                sorted(common)[0]
            times = {a: v[shape_key] for a, v in per_arm.items() if shape_key in v}
        else:
            shape_key = "(无共同形状，各取最大档)"
            def biggest(d):
                return max(d.values())
            times = {a: biggest(v) for a, v in per_arm.items()}
        if not times:
            continue
        defn = _definition_for(task)
        if not defn or defn.get("kind") != "kernel":
            continue
        # 用第一档 public 的 axes 算 roofline 地板
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            td = _task_dir_of(task, root)
            if td is None:
                continue
            d_obj = defn_mod.load(os.path.join(td, "definition.json"))
            from bench.data import workload as wl_mod
            pub, _ = wl_mod.split(wl_mod.load(os.path.join(td, "workload.jsonl"), d_obj))
            axes = dict(d_obj.const_axes()); axes.update(pub[0].axes)
            scores = perf_mod.rank(times, d_obj, axes)
        except Exception as e:
            out[task] = {"error": f"{type(e).__name__}: {e}"}
            continue
        entry: dict = {"_shape": shape_key}
        for arm, s in scores.items():
            entry[arm] = {
                "t_ms": round(s.t_ms, 5),
                "rel_ratio": round(s.rel_ratio, 4),
                "roof_ratio": round(s.roof_ratio, 4),
                "floor_ms": round(s.floor_ms, 5),
                "floor_kind": s.floor_kind,
                "note": s.note,
            }
        out[task] = entry
    return out


def _pct(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals) * 100.0


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _deltas(table: dict, tasks: list[str]) -> dict:
    """Δ_eff / Δ_halluc，以及每个臂的分布。

    只对**同一批题**比较：某臂缺一道题就不能直接拿比例相减，
    否则比的是"谁跑完了"，不是"谁答得好"。缺题时 pairs 里会标出来。
    """
    out: dict = {}
    for arm in ARMS:
        rows = [table[(t, arm)] for t in tasks if (t, arm) in table]
        out[arm] = {
            "n": len(rows),
            "eff_pct": _pct(rows, "effective"),
            "halluc_mean": _mean(rows, "halluc_rate"),
            "judged_mean": _mean(rows, "judged"),
            "leaked": sum(1 for r in rows if r["leaked"]),
        }

    def diff(a: str, b: str, key: str):
        x, y = out[a].get(key), out[b].get(key)
        if x is None or y is None:
            return None
        return round(x - y, 2)

    out["d_eff_C_minus_A"] = diff("C", "A", "eff_pct")
    out["d_eff_B_minus_A"] = diff("B", "A", "eff_pct")
    out["d_eff_C_minus_B"] = diff("C", "B", "eff_pct")
    out["d_halluc_C_minus_A"] = diff("C", "A", "halluc_mean")
    out["d_halluc_B_minus_A"] = diff("B", "A", "halluc_mean")

    # 胜负线（docs/scoring.md 第五节，跑前定死）
    verdicts = []
    de = out.get("d_eff_C_minus_A")
    dcb = out.get("d_eff_C_minus_B")
    dh = out.get("d_halluc_C_minus_A")
    if de is not None and de < 20:
        verdicts.append("Δ_eff(C−A) < 20pt ⇒ 该砍文档，不是加文档")
    if dcb is not None and abs(dcb) < 10:
        verdicts.append("B 与 C 差 < 10pt ⇒ 判据层无价值，只留库清单")
    if dh is not None and dh > 0:
        verdicts.append("Δ_halluc(C−A) > 0 ⇒ 立刻停，skill 在教模型编答案")
    if abs(out["A"]["n"] - out["C"]["n"]) > 0 or abs(out["B"]["n"] - out["C"]["n"]) > 0:
        verdicts.append(
            "三臂题数不一致，Δ 只能当参考：比例相减比的是覆盖率，不是能力"
        )
    out["verdicts"] = verdicts
    out["balanced"] = out["A"]["n"] == out["B"]["n"] == out["C"]["n"] > 0
    return out


def render_report(table: dict, template: str | None = None) -> str:
    tasks = table["tasks"]
    rows = table["table"]
    d = table["deltas"]
    lines = []
    lines.append("# toolkit-opbench 报告（三臂对照）")
    lines.append("")
    lines.append("## 结论一句话")
    lines.append("")
    if not d.get("balanced"):
        lines.append(
            "> 三臂题数不一致或尚未全跑，下面的 Δ 只能当参考，不能当结论。"
        )
    elif d.get("verdicts"):
        lines.append("> " + "；".join(d["verdicts"]))
    else:
        lines.append("> 尚未触发任何胜负线（样本量 " + str(d["A"]["n"]) + " 题/臂）。")
    lines.append("")

    lines.append("## 三臂对比")
    lines.append("")
    lines.append("| 题 | 臂 | 状态 | 有效结论 | judgment | 幻觉率 | 泄漏 | 墙钟 s | 产物 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for arm in ARMS:
            r = rows.get((t, arm))
            if not r:
                lines.append(f"| {t} | {arm} | 未跑 | — | — | — | — | — | — |")
                continue
            eff = "✓" if r["effective"] else "✗"
            judge = "—" if r["judged"] is None else str(r["judged"])
            lines.append(
                f"| {t} | {arm} | {r['status']} | {eff} | {judge} | "
                f"{r['halluc_rate']:.2f} | {'**命中**' if r['leaked'] else '0'} | "
                f"{r['wall_time_s']} | {r['artifacts']} |"
            )
    lines.append("")

    lines.append("## Δ")
    lines.append("")
    lines.append("| 指标 | A | B | C |")
    lines.append("|---|---|---|---|")
    for key, label in (("n", "题数"), ("eff_pct", "有效结论率 %"),
                       ("judged_mean", "judgment 均分"),
                       ("halluc_mean", "幻觉率")):
        vals = [d[a].get(key) for a in ARMS]
        fmt = lambda v: "—" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
        lines.append(f"| {label} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |")
    lines.append("")
    for key, label in (("d_eff_C_minus_A", "Δ_eff C−A (pt)"),
                       ("d_eff_B_minus_A", "Δ_eff B−A (pt)"),
                       ("d_eff_C_minus_B", "Δ_eff C−B (pt)"),
                       ("d_halluc_C_minus_A", "Δ_halluc C−A"),
                       ("d_halluc_B_minus_A", "Δ_halluc B−A")):
        v = d.get(key)
        lines.append(f"- {label}: " + ("—" if v is None else f"{v:+.2f}"))
    lines.append("")

    # ---- 性能（80% 权重）：主表 ----
    perf = table.get("perf") or {}
    if perf:
        lines.append("## 性能（bench 主指标，权重 80%）")
        lines.append("")
        lines.append("> 正确性是**硬门槛**：不通过该题的臂不进这张表。")
        lines.append("> 相对达成 = 本批最快 / 本臂；roofline = 地板 / 本臂。")
        lines.append("")
        lines.append("| 题 | 形状 | 臂 | 耗时 ms | 相对达成 | roofline | 地板 ms | 支配项 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for task, entry in perf.items():
            if "error" in entry:
                lines.append(f"| {task} | — | — | — | — | — | — | {entry['error']} |")
                continue
            shape = entry.get("_shape", "")
            for arm in ARMS:
                r = entry.get(arm)
                if not isinstance(r, dict):
                    continue
                lines.append(
                    f"| {task} | {shape} | {arm} | {r['t_ms']:.5f} | "
                    f"{r['rel_ratio']*100:.1f}% | {r['roof_ratio']*100:.1f}% | "
                    f"{r['floor_ms']:.5f} | {r['floor_kind']} |"
                )
        lines.append("")
        notes = [(t, a, r["note"]) for t, e in perf.items()
                 if isinstance(e, dict) for a, r in e.items()
                 if isinstance(r, dict) and r.get("note")]
        if notes:
            lines.append("**口径警告**（这些数先核对计时方式再用）：")
            for t, a, n in notes:
                lines.append(f"- {t} / {a}：{n}")
            lines.append("")

    lines.append("## 逐题判定理由")
    lines.append("")
    for t in tasks:
        for arm in ARMS:
            r = rows.get((t, arm))
            if not r:
                continue
            reasons = []
            reasons.append(f"有效结论：{r['effective_why']}")
            if r["judged_why"]:
                reasons.append(f"judgment：{r['judged_why']}")
            if r["leaked"]:
                reasons.append(f"泄漏命中：{r['leaks']}")
            lines.append(f"- **{t} / {arm}** — " + "；".join(reasons))
    lines.append("")

    lines.append("## 泄漏与污染")
    lines.append("")
    leaked = [(t, a) for t in tasks for a in ARMS
              if (t, a) in rows and rows[(t, a)]["leaked"]]
    if leaked:
        lines.append(f"- A 臂泄漏命中 {len(leaked)} 处，对应单元作废：{leaked}")
    else:
        lines.append("- 未命中泄漏指纹。")
    lines.append("- 时钟：见 runs/trace.jsonl 的 sclk_mhz（L5 题不碰 GPU，为空）")
    lines.append("")
    return "\n".join(lines)


def write_report(runs_dir: str, out_dir: str,
                 isolated_root: str | None = None,
                 only_tasks: list[str] | None = None) -> str:
    data = collect(runs_dir, isolated_root, only_tasks=only_tasks)
    text = render_report(data)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {**{k: v for k, v in data.items() if k != "table"},
             "table": {f"{t}/{a}": r for (t, a), r in data["table"].items()}},
            f, ensure_ascii=False, indent=2,
        )
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="opbench-collect")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="只汇报这些题（不给则扫 runs/ 全部——会把历史评测也算进来）")
    p.add_argument("--print", action="store_true")
    a = p.parse_args()
    path = write_report(a.runs, a.out, only_tasks=a.tasks)
    if a.print:
        print(open(path, encoding="utf-8").read())
    else:
        print(path)
