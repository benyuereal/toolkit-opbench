"""L2 flush。

L2 = 8 MB。buffer 用 16 MB（2×），每次计时迭代前写一遍把缓存挤掉。
我们在 25 MB buffer 上量出过 4571 GB/s（比只读峰值高一倍），就是没 flush。
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _buffer(nbytes: int):
    raise NotImplementedError


def flush(nbytes: int = 16 * 1024 * 1024) -> None:
    raise NotImplementedError


def reset_persisting_l2() -> None:
    """若平台支持持久化 L2，这里清掉（不支持则 no-op）。"""
    return None
