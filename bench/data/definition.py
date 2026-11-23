"""definition.json 的 schema 与校验。

一个题目的"固定部分"：符号轴、张量规格、容差、参考实现入口。
不负责：形状展开（workload.py）、执行（harness/run.py）。
"""
from dataclasses import dataclass, field
from typing import Any, Literal

KINDS = ("kernel", "judgment")
TIERS = ("L1", "L2", "L3", "L4", "L5")
AXIS_TYPES = ("const", "var")


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


def load(path: str) -> Definition:
    """读 definition.json 并校验。校验不过抛 ValueError（带全部问题）。"""
    raise NotImplementedError


def validate(d: dict) -> list[str]:
    """返回问题列表，空表示通过。要查：必填字段、kind/tier 取值、
    kernel 必须有 reference 与 tolerance、judgment 必须有 expected_verdict。"""
    raise NotImplementedError
