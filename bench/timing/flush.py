"""L2 flush。

L2 = 8 MB。buffer 用 16 MB（2×），每次计时迭代前写一遍把缓存挤掉。
我们在 25 MB buffer 上量出过 4571 GB/s（比只读峰值高一倍），就是没 flush。

## 为什么"写完 buffer"就能挤掉被测数据

被测 kernel 的热数据留在 L2 里。写一遍 16 MB（2× L2）会让它整体被
新写的数据替换掉——不是"保证替换"，是"按容量把旧行挤出去"。所以
buffer 必须 >= 2× L2，且**必须是写**（读不产生新行）。

## 为什么必须紧接着被测调用

flush 之后再有任何一次内存访问（包括 `torch.empty` 这类分配，如果触发了
cudaMalloc 的回调），都会把一部分 buffer 行又挤掉、让新的热行进来。
所以 flush 与计时启动之间只允许放一次 event.record()。

## reset_persisting_l2

AMD 有个 persistent L2 window，被 pin 住的行不会被普通写挤掉。gfx938 上
torch 没有暴露接口（`cudaDeviceSetLimit` 那套在 HIP 里也没有对应 API），
所以这里是 no-op。**no-op 是有意的**：如果哪天平台支持了却没清，
flush 会静默失效，量出的带宽会偏高——那种情况下宁可报错也不要继续。
"""
from functools import lru_cache

DEFAULT_FLUSH_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=1)
def _buffer(nbytes: int):
    """按字节数缓存一个 1-D byte buffer。缓存它是因为每次迭代都分配
    会引入分配开销，而且可能触发 cudaMalloc 从而干扰缓存状态。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("flush 需要 CUDA/HIP 可用")
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def flush(nbytes: int = DEFAULT_FLUSH_BYTES) -> None:
    """写一遍 16 MB buffer 把 L2 挤掉。**是写，不是读。**

    用 `.fill_(0)` 而不是 `zero_()`：前者是单 kernel 的连续写，
    后者在部分后端会走 memset 的快速路径（可能绕过正常的内存通路）。
    """
    import torch

    if nbytes < 2 * 8 * 1024 * 1024:
        raise ValueError(
            f"flush buffer 只有 {nbytes / 1e6:.1f} MB，小于 2×L2(8MB)。"
            "小 buffer 挤不掉热数据，量出的带宽会虚高——4571 GB/s 那次就是这么来的"
        )
    buf = _buffer(int(nbytes))
    buf.fill_(0)
    # 不需要 synchronize：后续 kernel 在同一 stream 上自然排队。
    # 在这里 sync 会把 flush 的耗时算进下一次计时之外的等待，没必要。


def reset_persisting_l2() -> None:
    """若平台支持持久化 L2，这里清掉（不支持则 no-op）。

    gfx938 + torch 2.11 没有可用接口，见模块 docstring。
    """
    return None


def is_supported() -> dict:
    """体检用：flush 能不能跑、buffer 多大、平台是否支持持久化 L2。"""
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "buffer_bytes": DEFAULT_FLUSH_BYTES,
        "persisting_l2_reset": False,
        "note": "gfx938 无持久化 L2 接口；若将来支持需在此实现，否则 flush 会静默失效",
    }
