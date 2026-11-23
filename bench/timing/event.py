"""用 GPU event 计时。禁用 wall clock。

异步 kernel 用 time.time() 会被量成 0——这是最基本也最容易犯的错。
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Timing:
    median_ms: float
    mad_ms: float
    cv: float
    samples: list[float]
    sclk_mhz: int | None
    warmup: int
    iters: int


def time_call(fn: Callable[[], None], warmup: int = 10, iters: int = 50) -> Timing:
    """固定 warmup 与 iters，逐次 event 计时，报 median + MAD。
    每次迭代前要 flush L2（调 flush.flush()），否则热数据虚高。"""
    raise NotImplementedError


def time_many(fns: dict[str, Callable], warmup: int = 10, iters: int = 50) -> dict[str, Timing]:
    """多个实现同一形状下对比。交替执行，避免温度漂移只影响一方。"""
    raise NotImplementedError
