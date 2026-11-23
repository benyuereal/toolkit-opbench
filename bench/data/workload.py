"""workload.jsonl：形状实例与输入生成。

一行一个实例：axes 取值 + inputs 生成方式 + split(public/holdout)。
holdout 只在评分时用，不能出现在题面里。
"""
from dataclasses import dataclass


@dataclass
class Workload:
    uuid: str
    axes: dict
    inputs: dict
    split: str = "public"

    def is_holdout(self) -> bool:
        return self.split == "holdout"


def load(path: str) -> list[Workload]:
    raise NotImplementedError


def split(ws: list[Workload]) -> tuple[list[Workload], list[Workload]]:
    """分成 (public, holdout)。至少各一，否则抛错。"""
    raise NotImplementedError


def build_inputs(w: Workload, d) -> dict:
    """按 inputs 声明造张量。要支持 type=random / zeros / scalar。
    生成必须可复现（固定 seed 由 uuid 派生）。"""
    raise NotImplementedError


def generate_holdout(d, n: int = 3, seed: int = 0) -> list[Workload]:
    """从 var 轴的取值域里采未出现过的形状。用于补 holdout。"""
    raise NotImplementedError
