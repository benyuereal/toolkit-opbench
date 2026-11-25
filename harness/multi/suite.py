"""读 suites/*.yaml。不依赖 PyYAML——这两个已知写法手写解析就够。

支持两种任务写法（仓库现存的两种混用）：

    tasks:
      - {id: moe-gating-l1, tier: L1, budget: 1800, needs_gpu: true}

    tasks:
      - id: gemm-fp8-l5
        family: gemm
        question: fp8 GEMM 在本机是否可用原生路径

顶层可给 `budget_per_task` 作为默认预算。

为什么不引 PyYAML：这两份文件是我们自己写的、结构固定；而 bench 要在
一个没装 yaml 的容器里也能跑。手写 40 行比多一个运行时依赖划算。
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_scalar(v: str):
    s = v.strip().strip('"').strip("'")
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        return s


def load_suite(name: str) -> list[dict]:
    path = os.path.join(ROOT, "suites", f"{name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"suite 不存在：{path}")

    tasks: list[dict] = []
    default_budget = None
    current: dict | None = None
    in_tasks = False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not line.startswith((" ", "\t")):
                in_tasks = s.startswith("tasks:")
                current = None
                if s.startswith("budget_per_task:"):
                    default_budget = int(s.split(":", 1)[1].strip())
                continue
            if not in_tasks:
                continue

            if s.startswith("- "):
                body = s[2:].strip()
                if body.startswith("{"):
                    d = {}
                    for part in body.strip("{}").split(","):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            d[k.strip()] = _parse_scalar(v)
                    current = {
                        "id": d.get("id"),
                        "budget": int(d["budget"]) if "budget" in d else default_budget,
                        "needs_gpu": bool(d.get("needs_gpu", True)),
                        "tier": d.get("tier"),
                    }
                    if current["id"]:
                        tasks.append(current)
                    current = None
                else:
                    k, _, v = body.partition(":")
                    current = {"id": _parse_scalar(v)}
                    tasks.append(current)
            elif current is not None and ":" in s:
                k, _, v = s.partition(":")
                k = k.strip()
                val = _parse_scalar(v)
                if k == "id":
                    current["id"] = val
                elif k == "budget":
                    current["budget"] = int(val)
                elif k == "needs_gpu":
                    current["needs_gpu"] = bool(val)
                elif k == "tier":
                    current["tier"] = val

    for t in tasks:
        t.setdefault("budget", default_budget)
        t.setdefault("needs_gpu", True)
        t.setdefault("tier", None)
    return [t for t in tasks if t.get("id")]
