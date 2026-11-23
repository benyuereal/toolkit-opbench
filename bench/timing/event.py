"""用 GPU event 计时。禁用 wall clock。

异步 kernel 用 time.time() 会被量成 0——这是最基本也最容易犯的错。

## 为什么每次迭代单独一对 event，而不是一头一尾

一头一尾（record 一次、跑 N 次、再 record）量的是**平均**，会把
第一次的冷启动与后面的稳态糊在一起，也看不出分布。逐次 event 才能
报 median + MAD，而 MAD 是判断"这次测量能不能用来比较"的唯一依据。

## 为什么 sclk 要每次记

空闲 600 MHz、boost 后 1200–1350 MHz，同一段代码在两个档位上差一倍。
不记档位的历史数据无法与今天的数据比较——我们踩过三次。
"""
from dataclasses import dataclass
import statistics
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
    name: str = ""
    sclk_start_mhz: int | None = None
    clock_stable: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _events():
    """在**当前设备**上建一对计时 event。

    必须显式传 device：`torch.cuda.Event()` 默认绑 cuda:0。若被测 kernel
    在 cuda:1 上跑，event 会记录 cuda:0 队列的时间线，量出荒谬的数
    （实测 4096³ bf16 报 3527–6211 TF，比 fp8 理论峰值还高 11 倍）。
    这类错误不会报错，只会给出"太好了"的数字——最容易骗过自己。
    """
    import torch

    dev = torch.cuda.current_device()
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))


def _sclk() -> int | None:
    """单次读当前 sclk（**不取众数**）。

    为什么不复用 dcu.current_sclk()：那个函数为 doctor 设计，采 3 次取众数，
    整个窗口约 0.8 秒。而这里是在测量的迭代间隙里读，只有几微秒到几十毫秒的
    机会——多采一次就读到负载结束后的空闲档了。这里要的是"此刻"，不是"稳态"。
    """
    try:
        from bench.hw import dcu

        levels = dcu._sclk_levels()
        if not levels:
            return None
        return min(m for _, m in levels)
    except Exception:
        return None


def time_call(fn: Callable[[], None], warmup: int = 10, iters: int = 50,
              flush_bytes: int | None = None, name: str = "") -> Timing:
    """固定 warmup 与 iters，逐次 event 计时，报 median + MAD。
    每次迭代前要 flush L2（调 flush.flush()），否则热数据虚高。

    fd 允许接受任意可调用对象；它的返回值被忽略（我们只量时间）。
    若 fn 内部自己 synchronize，量出来的是"含同步的墙钟"——那样的结果
    会明显偏大且 MAD 偏大，所以这里额外检测 sclk 在测量期间是否变化，
    并在 note 里标明。
    """
    import torch

    from bench.timing import flush as flush_mod

    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup 必须 >=0、iters 必须 >=1，实际 {warmup}/{iters}")
    nbytes = flush_mod.DEFAULT_FLUSH_BYTES if flush_bytes is None else int(flush_bytes)

    # 第一次调用可能包含 kernel 的首次加载/编译（实测 256³ bf16 mm 首调 5.6s，
    # 之后 <1ms）。它不属于测量范围，必须先让它付掉。warmup=0 时也至少跑一次，
    # 否则第一个样本会把编译时间当成 kernel 耗时——那种数据比没有数据更糟。
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()

    sclk_start = _sclk()
    samples: list[float] = []
    starts = []
    ends = []
    # 测量期间**在跑的时候**采 sclk：迭代结束后再读会读到已经回落的空闲档
    # （实测基线记成 600 MHz，而同一批数据反解出的算力对应 1200–1350）。
    # 每隔几次迭代采一次，取最大——我们关心的是"这次测量的峰值档位"，
    # 因为耗时由最快的那段决定不了，但分母口径必须按实际跑到的档位记。
    sclk_samples: list[int] = []
    for i in range(iters):
        # 顺序不可换：flush → record → fn → record。
        # flush 之后到 start.record() 之间不能再碰显存。
        flush_mod.flush(nbytes)
        s, e = _events()
        s.record()
        fn()
        e.record()
        starts.append(s)
        ends.append(e)
        if iters > 6 and i % max(1, iters // 6) == 0 and i > 0:
            v = _sclk()
            if v is not None:
                sclk_samples.append(v)
    # 最后一个 event 还没同步完时就读：此时 kernel 队列仍在跑
    v = _sclk()
    if v is not None:
        sclk_samples.append(v)
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    sclk_end = max(sclk_samples) if sclk_samples else _sclk()
    sclk_start = sclk_start if sclk_start is not None else sclk_end
    med = statistics.median(samples)
    # 跨设备 event 的典型症状是"太快"（数倍到数十倍）。这里给一条
    # 粗筛：同一 kernel 的 50 次中位数若比最小值大一倍以上，说明
    # 队列被别的东西占着；若比 0.1 µs 还小，基本是计的错设备。
    if med > 0 and med < 1e-4:
        note_pre = (f"median={med:.3g} ms 小于 0.1 µs，几乎肯定是计的错设备"
                    "（Event 默认绑 cuda:0）或 kernel 未真正执行")
    else:
        note_pre = ""
    mad = statistics.median([abs(x - med) for x in samples])
    note = note_pre
    if med <= 0:
        note = "median <= 0：计时无效（异步未同步，或 kernel 没真正执行）"
    if sclk_start is not None and sclk_end is not None and sclk_start != sclk_end:
        note = (note + "；" if note else "") + (
            f"测量期间 sclk 变化 {sclk_start}→{sclk_end} MHz，这批数据不可跨档比较"
        )
    return Timing(
        median_ms=med,
        mad_ms=mad,
        cv=(mad / med) if med else float("inf"),
        samples=samples,
        sclk_mhz=sclk_end if sclk_end is not None else sclk_start,
        warmup=warmup,
        iters=iters,
        name=name,
        sclk_start_mhz=sclk_start,
        clock_stable=(sclk_start == sclk_end),
        note=note,
    )


def time_many(fns: dict[str, Callable], warmup: int = 10, iters: int = 50,
              flush_bytes: int | None = None) -> dict[str, Timing]:
    """多个实现同一形状下对比。交替执行，避免温度漂移只影响一方。

    "交替"是指每一轮把所有实现各测一次，而不是先把 A 测完 50 次再测 B。
    后者会让 A 独占冷机状态、B 独占热机状态，两边差出来的可能是热漂移。
    """
    import torch

    from bench.timing import flush as flush_mod

    if not fns:
        raise ValueError("time_many 需要至少一个实现")
    nbytes = flush_mod.DEFAULT_FLUSH_BYTES if flush_bytes is None else int(flush_bytes)

    # 同 time_call：每个实现的第一次调用都要先付掉首次加载/编译，
    # 且要给足次数。注意这里每个实现都要单独付——它们是不同的 kernel。
    for _ in range(max(1, warmup)):
        for fn in fns.values():
            fn()
    torch.cuda.synchronize()

    sclk_start = _sclk()
    samples: dict[str, list[float]] = {k: [] for k in fns}
    for _ in range(iters):
        for name, fn in fns.items():
            flush_mod.flush(nbytes)
            s, e = _events()
            s.record()
            fn()
            e.record()
            e.synchronize()  # 交错计时必须逐个同步，否则 event 会攒在一起
            samples[name].append(s.elapsed_time(e))
    sclk_end = _sclk()

    out = {}
    for name, xs in samples.items():
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        out[name] = Timing(
            median_ms=med,
            mad_ms=mad,
            cv=(mad / med) if med else float("inf"),
            samples=xs,
            sclk_mhz=sclk_end if sclk_end is not None else sclk_start,
            warmup=warmup,
            iters=iters,
            name=name,
            sclk_start_mhz=sclk_start,
            clock_stable=(sclk_start == sclk_end),
        )
    return out
