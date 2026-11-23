"""definition.json 的 schema 与校验。

一个题目的"固定部分"：符号轴、张量规格、容差、参考实现入口。
不负责：形状展开（workload.py）、执行（harness/run.py）。

校验为什么要严格到"未知字段也报错"：题面是评分器的唯一输入，
一个字段名打错（tolerance -> tolerence）不会让加载失败，只会让容差
静默变成"没写"，然后按默认值放宽——这正是我们最想避免的那类事故。
"""
from dataclasses import dataclass, field
import json
from typing import Any, Literal

KINDS = ("kernel", "judgment")
TIERS = ("L1", "L2", "L3", "L4", "L5")
AXIS_TYPES = ("const", "var")

# torch dtype 名的白名单。不 import torch：data 层要能在无 GPU 的机器上跑单测。
DTYPES = (
    "float32", "float16", "bfloat16", "float64",
    "float8_e4m3fn", "float8_e5m2",
    "int8", "int16", "int32", "int64",
    "uint8", "bool",
)

# 每类题允许出现的顶层字段。多了就是拼错或抄错。
_COMMON_KEYS = {
    "id", "kind", "family", "tier", "precision", "axes", "inputs", "outputs",
    "tolerance", "reference", "dcu", "question", "notes",
}
_JUDGMENT_KEYS = {
    "id", "kind", "family", "tier", "question", "expected_verdict",
    "verdict_keywords", "forbidden_claims", "evidence_required", "notes",
}


@dataclass
class Definition:
    id: str
    kind: Literal["kernel", "judgment"]
    family: str
    tier: str
    axes: dict[str, dict]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    tolerance: dict | None = None
    reference: str | None = None
    dcu: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def const_axes(self) -> dict[str, int]:
        return {k: v["value"] for k, v in self.axes.items() if v["type"] == "const"}

    def var_axes(self) -> list[str]:
        return [k for k, v in self.axes.items() if v["type"] == "var"]

    def expected_verdict(self) -> str | None:
        return self.raw.get("expected_verdict")

    def verdict_keywords(self) -> list[str]:
        return list(self.raw.get("verdict_keywords") or [])

    def forbidden_claims(self) -> list[str]:
        return list(self.raw.get("forbidden_claims") or [])

    def evidence_required(self) -> list[str]:
        return list(self.raw.get("evidence_required") or [])


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_axis(name: str, spec: Any, problems: list[str]) -> None:
    if not isinstance(spec, dict):
        problems.append(f"axes.{name} 必须是对象")
        return
    t = spec.get("type")
    if t not in AXIS_TYPES:
        problems.append(f"axes.{name}.type 必须是 {AXIS_TYPES} 之一，实际 {t!r}")
        return
    if t == "const":
        v = spec.get("value")
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            problems.append(f"axes.{name}.value 必须是正整数，实际 {v!r}")
        extra = set(spec) - {"type", "value"}
        if extra:
            problems.append(f"axes.{name} 是 const，不该有字段 {sorted(extra)}")
    else:  # var
        extra = set(spec) - {"type", "domain"}
        if extra:
            problems.append(f"axes.{name} 是 var，不该有字段 {sorted(extra)}")
        dom = spec.get("domain")
        if dom is not None:
            if not isinstance(dom, dict):
                problems.append(f"axes.{name}.domain 必须是对象")
            else:
                vals = dom.get("values")
                if vals is not None:
                    if not isinstance(vals, list) or not vals:
                        problems.append(f"axes.{name}.domain.values 必须是非空数组")
                    for v in vals or []:
                        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                            problems.append(
                                f"axes.{name}.domain.values 含非正整数 {v!r}"
                            )
                else:
                    lo, hi = dom.get("min"), dom.get("max")
                    if not (isinstance(lo, int) and isinstance(hi, int) and lo > 0 and hi >= lo):
                        problems.append(
                            f"axes.{name}.domain 需要 values 或 min/max 正整数"
                        )


def _check_tensor(name: str, spec: Any, axis_names: set[str], problems: list[str]) -> None:
    if not isinstance(spec, dict):
        problems.append(f"{name} 必须是对象")
        return
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape:
        problems.append(f"{name}.shape 必须是非空数组")
    else:
        for dim in shape:
            if not isinstance(dim, str):
                problems.append(f"{name}.shape 的元素必须是轴名字符串，实际 {dim!r}")
            elif dim not in axis_names:
                problems.append(f"{name}.shape 引用了未声明的轴 {dim!r}")
    dt = spec.get("dtype")
    if dt not in DTYPES:
        problems.append(f"{name}.dtype {dt!r} 不在白名单 {DTYPES}")


def _validate_peak_id(val: Any, problems: list[str]) -> None:
    """dcu.peak_constant 只许写 constants 里的 id，不许写字面量（560 这类）。"""
    if val is None:
        return
    if not isinstance(val, str):
        problems.append("dcu.peak_constant 必须是字符串 id，不许是数值字面量")
        return
    # 峰值写进 id 的形态：以 _<三位以上数字> 结尾，如 fp8_peak_nominal_560。
    # 不能一概而论地"含数字就拒"——上游 id 里本来就有 bf16 / fp8。
    tail = val.rsplit("_", 1)[-1]
    if tail.isdigit() and len(tail) >= 3:
        problems.append(
            f"dcu.peak_constant={val!r} 把峰值写进了 id；"
            "只许引用 constants 的 id（如 peak_achievable:bf16）"
        )
        return
    try:
        from bench.hw import constants as hw_constants
    except ImportError as e:  # pragma: no cover - 只在缺 PyYAML 时触发
        problems.append(f"无法校验 peak_constant（{e}）")
        return
    try:
        hw_constants.get(val)
    except KeyError as e:
        problems.append(f"dcu.peak_constant 引用了不存在的常数 id：{e}")
    except hw_constants.ConstantsError as e:
        problems.append(f"无法校验 dcu.peak_constant：{e}")


def validate(d: dict) -> list[str]:
    """返回问题列表，空表示通过。要查：必填字段、kind/tier 取值、
    kernel 必须有 reference 与 tolerance、judgment 必须有 expected_verdict。"""
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["definition 必须是 JSON 对象"]

    kind = d.get("kind")
    if kind not in KINDS:
        problems.append(f"kind 必须是 {KINDS} 之一，实际 {kind!r}")

    tier = d.get("tier")
    if tier not in TIERS:
        problems.append(f"tier 必须是 {TIERS} 之一，实际 {tier!r}")

    for key in ("id", "family"):
        v = d.get(key)
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{key} 必须是非空字符串")

    allowed = _JUDGMENT_KEYS if kind == "judgment" else _COMMON_KEYS
    unknown = set(d) - allowed
    if unknown:
        problems.append(f"未知字段 {sorted(unknown)}（拼错？字段必须来自 docs/format.md）")

    if kind == "judgment":
        if not isinstance(d.get("question"), str) or not d["question"].strip():
            problems.append("judgment 必须有 question")
        ev = d.get("expected_verdict")
        if ev not in ("available", "unavailable", "conditional", "unknown"):
            problems.append(
                "judgment 必须有 expected_verdict ∈ "
                "available/unavailable/conditional/unknown"
            )
        for key in ("verdict_keywords", "forbidden_claims"):
            v = d.get(key)
            if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
                problems.append(f"judgment 必须有非空的字符串数组 {key}")
        er = d.get("evidence_required")
        if not isinstance(er, list) or not er or not all(isinstance(x, str) and x for x in er):
            problems.append("judgment 必须有非空的字符串数组 evidence_required")
        return problems

    # ---- kernel ----
    axes = d.get("axes")
    if not isinstance(axes, dict) or not axes:
        problems.append("axes 必须是非空对象")
        axes = {}
    else:
        for name, spec in axes.items():
            _check_axis(name, spec, problems)

    axis_names = set(axes)
    for key in ("inputs", "outputs"):
        obj = d.get(key)
        if not isinstance(obj, dict) or not obj:
            problems.append(f"{key} 必须是非空对象")
            continue
        for name, spec in obj.items():
            _check_tensor(f"{key}.{name}", spec, axis_names, problems)

    tol = d.get("tolerance")
    if not isinstance(tol, dict):
        problems.append("kernel 必须有 tolerance")
    else:
        extra = set(tol) - {"rtol", "atol"}
        if extra:
            problems.append(f"tolerance 只认 rtol/atol，多了 {sorted(extra)}")
        for key in ("rtol", "atol"):
            if not _is_num(tol.get(key)):
                problems.append(f"tolerance.{key} 必须是数值")

    ref = d.get("reference")
    if not isinstance(ref, str) or not ref.strip():
        problems.append("kernel 必须有 reference（同目录脚本，暴露 arun）")

    precision = d.get("precision")
    if not isinstance(precision, list) or not precision:
        problems.append("kernel 必须有非空的 precision 数组")
    else:
        for p in precision:
            if p not in DTYPES:
                problems.append(f"precision 含未知 dtype {p!r}")

    dcu = d.get("dcu")
    if dcu is not None:
        if not isinstance(dcu, dict):
            problems.append("dcu 必须是对象")
        else:
            _validate_peak_id(dcu.get("peak_constant"), problems)
            mb = dcu.get("min_l2_flush_mb")
            if mb is not None and (not _is_num(mb) or mb <= 0):
                problems.append("dcu.min_l2_flush_mb 必须是正数")
    return problems


def load(path: str) -> Definition:
    """读 definition.json 并校验。校验不过抛 ValueError（带全部问题）。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    problems = validate(d)
    if problems:
        raise ValueError(
            f"{path} 校验失败（{len(problems)} 项）：\n  - " + "\n  - ".join(problems)
        )
    return Definition(
        id=d["id"],
        kind=d["kind"],
        family=d["family"],
        tier=d["tier"],
        axes=d.get("axes") or {},
        inputs=d.get("inputs") or {},
        outputs=d.get("outputs") or {},
        tolerance=d.get("tolerance"),
        reference=d.get("reference"),
        dcu=d.get("dcu") or {},
        raw=d,
    )
