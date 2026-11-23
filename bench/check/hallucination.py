"""幻觉检测：它说的函数/库/数字，是否真的存在。

这是我们历史上最贵的错：in-house 写了 29 条、两个幽灵库、49.8 当峰值。

## 判法上的两个取舍

* **import 只验到"模块存在 + 属性存在"这一层。** 不真的 import 重型库
  （lightop/aiter 会初始化 HCC 运行时）。代价是"函数存在但签名不对"
  查不出来——那类错误由正确性检查兜底，不属于幻觉。
* **库名对照 canonical_id 白名单，但对 `installed: false` 的条目另算。**
  hipBLASLt / TileKernels 在 registry 里存在（带 not_installed_reason），
  声称"用 hipBLASLt 做的"不算编造库名，算**编造已安装**——两者都记失败，
  但理由不同，报告里要分开说。
"""
from __future__ import annotations

import importlib
import json
import os
import re


def verify_import_path(path: str) -> tuple[bool, str]:
    """逐条 importlib.import_module + hasattr。失败返回原因。

    两种情况都算"存在"，缺一不可：
      1. **子模块**：`aiter.fused_moe` 不是一个属性，是一个可导入模块。
         只查 hasattr 会把它判成编造——实测这会让 C 臂平白多两条"幻觉"。
      2. **属性链**：`aiter.asm_fmoe_a8` 是模块上的属性。
    从最长的前缀往下试，先试子模块导入，再试属性链。
    """
    if not path or "." not in path:
        return False, "不是一个点分路径"
    parts = path.split(".")

    # 1) 整条路径可否直接 import 成模块（覆盖 aiter.fused_moe 这类）
    for i in range(len(parts), 0, -1):
        mod_path = ".".join(parts[:i])
        try:
            importlib.import_module(mod_path)
        except Exception:
            continue
        rest = parts[i:]
        if not rest:
            return True, "存在（模块）"
        # 剩下的部分必须是属性链
        obj = importlib.import_module(mod_path)
        for a in rest:
            if not hasattr(obj, a):
                return False, f"{mod_path} 里没有 {'.'.join(rest)}"
            obj = getattr(obj, a)
        return True, "存在"
    return False, f"无法导入 {path} 的任何前缀模块"


def _registry(path: str) -> dict:
    """canonical_id → 条目。`removed` 段的条目也并入，但标 `_removed`。

    为什么并入：那两个幽灵库（hipBLASLt / TileKernels）在 registry 里
    是"明确不在环境里"的记录，不是"没听说过"。分开记才能给出准确理由
    ——「编造了一个不存在的库」和「用了已移除的库」是两种错，
    报告里要分开说，但**都是失败**。
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return {}
    out = {}
    for lib in data.get("libraries") or []:
        cid = lib.get("canonical_id")
        if cid:
            out[cid] = lib
    for lib in data.get("removed") or []:
        cid = lib.get("canonical_id")
        if cid:
            out[cid] = {**lib, "_removed": True, "installed": False}
    return out


_ALIAS_NORM = str.maketrans("_-", "..")


def verify_library(name: str, registry_path: str) -> bool:
    """库名必须在 libraries-registry.yaml 的 canonical_id 白名单里。

    别名也认（registry 里有 aliases 字段）：同一个库有多个异名并存，
    正是我们加 aliases 的原因。
    """
    reg = _registry(registry_path)
    if not reg:
        return False
    key = (name or "").strip().lower()
    if key in reg:
        return True
    for cid, lib in reg.items():
        for alias in lib.get("aliases") or []:
            if str(alias).strip().lower() == key:
                return True
        imp = (lib.get("import_name") or "").lower()
        if imp and imp == key:
            return True
    return False


def verify_library_installed(name: str, registry_path: str) -> tuple[bool, str]:
    """声称用某库时，该库必须 installed: true。

    区分两种错：
      * 库名不在表里 → 编造库名（幽灵库）
      * 在表里但 installed: false → 编造"装了"。hipBLASLt 那次是后者。
    """
    reg = _registry(registry_path)
    key = (name or "").strip().lower()
    entry = reg.get(key)
    if entry is None:
        for cid, lib in reg.items():
            names = [str(a).lower() for a in (lib.get("aliases") or [])]
            names.append((lib.get("import_name") or "").lower())
            if key in names:
                entry = lib
                break
    if entry is None:
        return False, "不在 libraries-registry 里（编造库名）"
    if entry.get("_removed"):
        reason = entry.get("reason") or "已从环境移除"
        repl = entry.get("replaced_by")
        hint = f"（可替代：{repl}）" if repl else ""
        return False, f"该库已从环境移除{hint}：{reason}"
    if not entry.get("installed"):
        # registry 里有些条目根本没有 installed 字段，但带着 reason
        # （"不是可导入包"/"import 失败"）——那也是不可用。
        reason = (entry.get("not_installed_reason")
                  or entry.get("reason") or "未安装")
        return False, f"该库不可用：{reason}"
    return True, "已装"


def verify_numbers_against_results(claims: dict, results: dict) -> list[str]:
    """它报的数字必须能在 results/*.json 里找到。找不到就是编的。

    容差：相对 1e-3。数字在不同单位/四舍五入下会有末位差，
    但不接受"量级对不上"——那正是 49.8 那次的问题。
    """
    problems = []
    flat = _flatten_numbers(results)
    for item in (claims or {}).get("numbers") or []:
        v = item.get("value")
        if v is None:
            continue
        if any(abs(v - r) <= max(abs(v), abs(r)) * 1e-3 for r in flat):
            continue
        problems.append(
            f"{item.get('kind')}={v}（出处 {item.get('source')}）在 results 里找不到"
        )
    return problems


def _flatten_numbers(obj, out=None) -> list[float]:
    if out is None:
        out = []
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


def _load_results(workdir: str) -> dict:
    rdir = os.path.join(workdir, "results")
    out = {}
    if not os.path.isdir(rdir):
        return out
    for name in sorted(os.listdir(rdir)):
        if not name.endswith((".json", ".jsonl")):
            continue
        p = os.path.join(rdir, name)
        try:
            with open(p, encoding="utf-8") as f:
                if name.endswith(".jsonl"):
                    out[name] = [json.loads(x) for x in f if x.strip()]
                else:
                    out[name] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def check_solution(solution, workdir: str, registry_path: str) -> dict:
    """汇总：{checked: n, failed: [(claim, reason)], rate: float}

    solution 若为 None，从 workdir 的产物里抽（评分器的常规用法：
    只读产物，不信自述）。
    """
    from bench.data import solution as sol_mod

    claims = sol_mod.extract_claims(workdir)
    if solution is not None and getattr(solution, "claims", None):
        claims["numbers"] = list(claims.get("numbers") or [])
        for key in ("ratio", "speedup"):
            v = solution.claims.get(key)
            if isinstance(v, (int, float)):
                claims["numbers"].append({"kind": key, "value": float(v),
                                          "source": "solution.json"})

    results = _load_results(workdir)
    failed: list[tuple[str, str]] = []
    checked = 0

    for path in claims.get("import_paths") or []:
        checked += 1
        ok, why = verify_import_path(path)
        if not ok:
            failed.append((path, why))

    for lib in claims.get("libraries") or []:
        checked += 1
        ok, why = verify_library_installed(lib, registry_path)
        if not ok:
            failed.append((f"lib:{lib}", why))

    for problem in verify_numbers_against_results(claims, results):
        checked += 1
        failed.append(("number", problem))

    return {
        "checked": checked,
        "failed": failed,
        "rate": (len(failed) / checked) if checked else 0.0,
        "claims": claims,
    }
