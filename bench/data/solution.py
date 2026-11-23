"""solution.json：agent 提交的答案，以及产物清单校验。

kernel 类：要交可执行入口 + 跑出来的 results/
judgment 类：要交结论 + 依据 + 复跑命令

为什么 kernel 类必须交 solution.json 而不能"看 report.md 里的数字"：
report.md 是散文，散文里的数字可以挑好看的写。solution.json 是
评分器的输入契约，它的 claims 会被逐条对着结果文件与 import 空间验。
"""
from dataclasses import dataclass
import json
import os
import re

# 判定题必须有的证据段落（agent 侧写进 report.md，机检时按标题或行内标记认）
JUDGMENT_EVIDENCE_MARKERS = ("证据", "依据", "evidence", "复跑", "reproduce", "命令")


@dataclass
class Solution:
    task_id: str
    kind: str
    entrypoint: str | None = None
    claims: dict | None = None       # 结论/数字/引用的函数
    notes: str | None = None


def validate(d: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["solution 必须是 JSON 对象"]
    tid = d.get("task_id")
    if not isinstance(tid, str) or not tid.strip():
        problems.append("task_id 必须是非空字符串")
    kind = d.get("kind")
    if kind not in ("kernel", "judgment"):
        problems.append("kind 必须是 kernel 或 judgment")
    entry = d.get("entrypoint")
    if kind == "kernel":
        if not isinstance(entry, str) or ":" not in entry:
            problems.append("kernel 类必须有 entrypoint，形如 'solution.py:run'")
    claims = d.get("claims")
    if claims is not None and not isinstance(claims, dict):
        problems.append("claims 必须是对象")
    return problems


def load(path: str) -> Solution:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    problems = validate(d)
    if problems:
        raise ValueError(
            f"{path} 校验失败（{len(problems)} 项）：\n  - " + "\n  - ".join(problems)
        )
    return Solution(
        task_id=d["task_id"],
        kind=d["kind"],
        entrypoint=d.get("entrypoint"),
        claims=d.get("claims"),
        notes=d.get("notes"),
    )


def _result_files(workdir: str) -> list[str]:
    d = os.path.join(workdir, "results")
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json")
    )


def validate_artifacts(workdir: str, kind: str) -> list[str]:
    """校验产物是否齐全。kernel 要有 results/*.json 且含 timing 字段；
    judgment 要有 report.md 且含证据段落。返回问题列表。"""
    problems: list[str] = []
    if kind == "kernel":
        sol = os.path.join(workdir, "solution.json")
        if not os.path.isfile(sol):
            problems.append("缺 solution.json（评分器只读它，不读自述）")
        files = _result_files(workdir)
        if not files:
            problems.append("缺 results/*.json：没有结果文件直接判 0")
        else:
            for path in files:
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    problems.append(f"{os.path.relpath(path, workdir)} 读不了：{e}")
                    continue
                if not isinstance(data, dict):
                    problems.append(f"{os.path.relpath(path, workdir)} 必须是 JSON 对象")
                    continue
                if "timing" not in data and "t_ms" not in data:
                    problems.append(
                        f"{os.path.relpath(path, workdir)} 缺 timing/t_ms 字段："
                        "这是「可复跑数字」的载体"
                    )
    elif kind == "judgment":
        report = os.path.join(workdir, "report.md")
        if not os.path.isfile(report):
            problems.append("缺 report.md（判定题的结论载体）")
        else:
            with open(report, encoding="utf-8", errors="replace") as f:
                text = f.read()
            low = text.lower()
            if not any(m.lower() in low for m in JUDGMENT_EVIDENCE_MARKERS):
                problems.append(
                    "report.md 没有证据/复跑命令段落：判定题必须有可复跑的证据"
                )
        if not os.path.isdir(os.path.join(workdir, "scripts")):
            problems.append("缺 scripts/：证据要能复跑，不能只写结论")
        if not os.path.isdir(os.path.join(workdir, "results")):
            problems.append("缺 results/：原始输出（报错原文/日志）要落盘")
    else:
        problems.append(f"未知 kind {kind!r}")
    return problems


# 从散文里抽"声称存在的东西"。宁可多抽（幻觉检测多查一条不亏），
# 也不要漏抽——漏掉的每一条都是可以编造的空间。
_IMPORT_PATH = re.compile(
    r"\b((?:[a-zA-Z_][\w]*\.){1,6}[a-zA-Z_]\w*)\s*\("
)
_IMPORT_FROM = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", re.M)
_LIB_HINT = re.compile(
    r"\b(lightop|aiter|deepgemm|flash[_-]?attn|flash[_-]?mla|causal[_-]conv1d|"
    r"deep[_-]ep|boltops|tilelang|triton|sglang|sgl_kernel|lmslim|torch|"
    r"hipblaslt|tilekernels|hbird[_-]gemm)\b",
    re.I,
)
_RATIO = re.compile(r"(?:ratio|加速比|speedup)\s*(?:=|:|为|是)?\s*(\d+(?:\.\d+)?)\s*[x×]?", re.I)


# 报告/脚本里高频出现、且本来就该视为"库根名"的名字。
# 不在此表、又没被 import 过的根名一律不当成 import_path 声明。
_KNOWN_ROOTS = {
    "torch", "np", "numpy", "os", "sys", "json", "math", "re", "time",
    "aiter", "lightop", "deepgemm", "flash_attn", "flash_mla", "triton",
    "tilelang", "boltops", "sgl_kernel", "causal_conv1d", "deep_ep",
    "pathlib", "subprocess", "collections", "typing", "logging",
    "functional",  # torch.nn.functional 常被 from ... import 成 F
}

_IMPORT_LINE = re.compile(
    r"^\s*(?:import\s+([\w.]+)(?:\s+as\s+(\w+))?|from\s+([\w.]+)\s+import\s+(.+))",
    re.M,
)
# 任何位置的赋值，含元组解包 `a, b = ...`：代码块里的中间变量常这样出现，
# 只认行首会漏掉 `a8, a8s = make_quant(x)` 里的两个名字。
_ASSIGN = re.compile(
    r"(?:^|[\s,(])\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=\s*[^=]",
    re.M,
)


# 函数参数与循环变量：这两类名字一定是局部量，不可能是模块。
_LOCAL_BIND = re.compile(
    r"(?:def\s+\w+\s*\(([^)]*)\)|for\s+([\w,\s]+)\s+in\b)"
)


def _local_names(text: str) -> set[str]:
    """从 def 参数表与 for 循环头里收集局部名。"""
    out: set[str] = set()
    for m in _LOCAL_BIND.finditer(text):
        blob = m.group(1) or m.group(2) or ""
        for part in blob.split(","):
            name = part.strip().split("=")[0].strip().split(":")[0].strip()
            name = name.lstrip("*").strip()
            if name.isidentifier():
                out.add(name)
    return out


def _looks_like_module(name: str, assigned: set[str], local: set[str]) -> bool:
    """区分"像库/模块的名字"与"像局部变量的名字"。

    判错的代价不对称：
      * 把局部变量当模块 → 多查一条，浪费一次 import，无实质损失；
      * 把模块当局部变量 → **漏掉一个编造的库名**，正是要抓的东西。
    所以规则是"除非能证明它是局部变量，否则当模块查一遍"。
    """
    if name in local:
        return False          # def 参数 / for 变量：确定是局部量
    if name in assigned and name.islower():
        return False          # 被赋值过的纯小写名（a = ...、buf = ...）
    return True


def _binding_names(text: str) -> tuple[dict[str, str], set[str]]:
    """返回 (导入名 → 其来源模块, 被赋值过的名字)。

    `bound`：import x / import x as y / from m import a as b 引入的名字，
    映射到**它真正的模块路径**。别名必须归一：「from torch.nn import
    functional as F」之后出现的 `F.silu` 应验成 `torch.nn.functional.silu`，
    而不是去 import 一个叫 F 的模块。
    `assigned`：任何 `name = ...`。两者都有的名字（先 import 后重绑）
    以 import 为准——那通常是真的在用库。
    """
    bound: dict[str, str] = {}
    for m in _IMPORT_LINE.finditer(text):
        mod, alias, src, names = m.groups()
        if mod:                                    # import a.b  /  import a as b
            if alias:
                bound[alias] = mod
            else:
                bound[mod.split(".")[0]] = mod.split(".")[0]
        elif src:                                  # from m import a, b as c
            for part in names.split(","):
                part = part.strip().split(" as ")
                if not part or not part[0]:
                    continue
                bound[part[-1].strip()] = f"{src}.{part[0].strip()}"
    assigned: set[str] = set()
    for m in _ASSIGN.finditer(text):
        for part in m.group(1).split(","):
            name = part.strip()
            if name:
                assigned.add(name)
    return bound, assigned


def extract_claims(workdir: str) -> dict:
    """从产物里抽出"它声称的东西"：import_path、库名、数字、形状。
    交给 check/hallucination.py 逐条验。"""
    texts: list[tuple[str, str]] = []
    for name in ("report.md",):
        p = os.path.join(workdir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                texts.append((name, f.read()))
    sdir = os.path.join(workdir, "scripts")
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            p = os.path.join(sdir, name)
            if os.path.isfile(p) and name.endswith((".py", ".sh", ".md", ".txt")):
                with open(p, encoding="utf-8", errors="replace") as f:
                    texts.append((os.path.join("scripts", name), f.read()))

    import_paths: set[str] = set()
    libraries: set[str] = set()
    numbers: list[dict] = []
    # 局部名与导入名**跨文件合并**：报告里的代码块常常是脚本的节选，
    # 变量在脚本里定义、在报告里使用。按单文件判断会把 score.float 这类
    # 合法调用判成编造。
    all_bound: dict[str, str] = {}
    all_assigned: set[str] = set()
    all_local: set[str] = set()
    for _, text in texts:
        b, a = _binding_names(text)
        all_bound.update(b)
        all_assigned |= a
        all_local |= _local_names(text)

    for name, text in texts:
        # 已知的导入名（import x / from x import / 赋值给它们的名字）。
        # 只有"根名字确实是导入来的"才把 a.b 当成声称的 import_path；
        # 否则 a.float / P.make_quant 这类局部变量方法会被批量误判成
        # 编造——实测这会让幻觉率虚高到 0.6（三臂都中招）。
        bound, assigned = all_bound, all_assigned
        local = all_local
        for m in _IMPORT_PATH.finditer(text):
            path = m.group(1)
            root = path.split(".")[0]
            if root in ("self", "cls"):
                continue
            if root in assigned and root not in bound:
                continue          # 局部别名，如 a = x.abs()、P = probe
            if root in bound:
                # 别名归一：F.silu → torch.nn.functional.silu
                real = bound[root]
                path = real + path[len(root):]
                import_paths.add(path)
                continue
            if root in _KNOWN_ROOTS:
                import_paths.add(path)
                continue
            # 未知根名：**不能静默丢掉**。幽灵库（hipBLASLt / TileKernels）
            # 与编造的 import_path 恰恰长得就是这样——没被 import 过、
            # 也不在已知库里。丢掉它们等于把幻觉检测的主要目标筛没了。
            if _looks_like_module(root, assigned, local):
                import_paths.add(path)
        for m in _IMPORT_FROM.finditer(text):
            import_paths.add(m.group(1))
        for m in _LIB_HINT.finditer(text):
            libraries.add(m.group(1).lower().replace("-", "_"))
        for m in _RATIO.finditer(text):
            numbers.append({"kind": "ratio", "value": float(m.group(1)), "source": name})

    sol_path = os.path.join(workdir, "solution.json")
    if os.path.isfile(sol_path):
        try:
            sol = load(sol_path)
        except (ValueError, json.JSONDecodeError):
            sol = None
        if sol and sol.claims:
            c = sol.claims
            if isinstance(c.get("impl"), str):
                import_paths.add(c["impl"])
            for key in ("ratio", "speedup"):
                if isinstance(c.get(key), (int, float)):
                    numbers.append(
                        {"kind": key, "value": float(c[key]), "source": "solution.json"}
                    )

    return {
        "import_paths": sorted(p for p in import_paths if "." in p),
        "libraries": sorted(libraries),
        "numbers": numbers,
        "files_scanned": [n for n, _ in texts],
    }
