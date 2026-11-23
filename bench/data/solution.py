"""solution.json：agent 提交的答案，以及产物清单校验。

kernel 类：要交可执行入口 + 跑出来的 results/
judgment 类：要交结论 + 依据 + 复跑命令
"""
from dataclasses import dataclass


@dataclass
class Solution:
    task_id: str
    kind: str
    entrypoint: str | None = None
    claims: dict | None = None       # 结论/数字/引用的函数
    notes: str | None = None


def load(path: str) -> Solution:
    raise NotImplementedError


def validate_artifacts(workdir: str, kind: str) -> list[str]:
    """校验产物是否齐全。kernel 要有 results/*.json 且含 timing 字段；
    judgment 要有 report.md 且含证据段落。返回问题列表。"""
    raise NotImplementedError


def extract_claims(workdir: str) -> dict:
    """从产物里抽出"它声称的东西"：import_path、库名、数字、形状。
    交给 check/hallucination.py 逐条验。"""
    raise NotImplementedError
