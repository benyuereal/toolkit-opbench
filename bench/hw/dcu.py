"""DCU 探测：卡数、L2、当前 sclk、锁频能力、库版本。

实测事实（2026-11-23，gfx938）：
  - torch.cuda.Event 可用
  - L2 = 8 MB  → flush buffer 用 16 MB
  - 空闲 sclk 600 MHz，工作档 1150 MHz  → 测量前必须 ramp，并记录 sclk
  - hy-smi 没有 setclocks，只有 --save/--load  → 锁频要自己拼
"""
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    count: int
    name: str
    l2_bytes: int
    sclk_mhz: int | None
    busy_pct: list[float]


def query_devices() -> DeviceInfo:
    raise NotImplementedError


def current_sclk() -> int | None:
    """读 hy-smi -g。解析不到就返回 None（不要假装读到）。"""
    raise NotImplementedError


def is_busy(threshold: float = 20.0) -> bool:
    """任一卡 HCU% > threshold 即视为占用。GPU 串行的判断依据。"""
    raise NotImplementedError


def try_lock_clocks(level: int = 3) -> bool:
    """试 hy-smi --save / 改 / --load 锁频。失败返回 False，不抛异常。"""
    raise NotImplementedError


def ramp_to_work_clock(timeout_s: int = 10) -> int | None:
    """跑 dummy 负载直到 sclk 到工作档，返回最终 sclk。"""
    raise NotImplementedError


def library_versions() -> dict:
    """lightop / aiter / deepgemm / torch / DTK。不记录就无法复跑。"""
    raise NotImplementedError
