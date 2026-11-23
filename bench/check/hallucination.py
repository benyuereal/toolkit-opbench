"""幻觉检测：它说的函数/库/数字，是否真的存在。

这是我们历史上最贵的错：in-house 写了 29 条、两个幽灵库、49.8 当峰值。
"""
import importlib


def verify_import_path(path: str) -> tuple[bool, str]:
    """逐条 importlib.import_module + hasattr。失败返回原因。"""
    raise NotImplementedError


def verify_library(name: str, registry_path: str) -> bool:
    """库名必须在 libraries-registry.yaml 的 canonical_id 白名单里。"""
    raise NotImplementedError


def verify_numbers_against_results(claims: dict, results: dict) -> list[str]:
    """它报的数字必须能在 results/*.json 里找到。找不到就是编的。"""
    raise NotImplementedError


def check_solution(solution, workdir: str, registry_path: str) -> dict:
    """汇总：{checked: n, failed: [(claim, reason)], rate: float}"""
    raise NotImplementedError
