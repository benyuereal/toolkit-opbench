"""数值正确性：对参考实现逐 shape 对比，含 holdout 与压力数值。"""
from dataclasses import dataclass


@dataclass
class Result:
    ok: bool
    max_abs: float | None = None
    cos: float | None = None
    failed_workloads: list[str] | None = None
    note: str = ""


def compare(actual, expected, tol: dict) -> Result:
    """按 rtol/atol 比。tolerance 来自 definition，不许临时放宽。"""
    raise NotImplementedError


def run_reference(defn, workload, ref_module) -> object:
    raise NotImplementedError


def check_all(defn, workloads, solution_fn, ref_module) -> Result:
    """public + holdout + 压力数值（极端 dtype 取值，如 fp16 上溢）。"""
    raise NotImplementedError
