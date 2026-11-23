"""硬件常数的唯一来源。

**禁止在代码里写 560 / 310 这类字面量。** 一律按 id 从上游 constants.yaml 读，
原因是分母已经错过两次（540 无来源、49.8 当峰值），错一次结论全反。
"""
from functools import lru_cache

UPSTREAM = (
    "/workspace/toolkit_skills/Plugins/ops/skills/op-roofline/constants.yaml"
)
# 只允许引用这三类，不许混
KINDS = ("peak_nominal", "peak_achievable", "impl_measured")


@lru_cache(maxsize=1)
def load_all() -> dict:
    raise NotImplementedError


def get(const_id: str) -> dict:
    """返回 {value, unit, kind, status, source}。查不到抛 KeyError，
    不许 fallback 到默认值——那等于又编了一个数。"""
    raise NotImplementedError


def peak(kind: str, precision: str) -> float:
    """按 类别+精度 取峰值。kind 必须是 KINDS 之一。"""
    raise NotImplementedError
