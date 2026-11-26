"""DCU 探测：卡数、L2、当前 sclk、锁频能力、库版本。

实测事实（2026-11-23，gfx938）：
  - torch.cuda.Event 可用
  - L2 = 8 MB  → flush buffer 用 16 MB
  - 空闲 600 MHz，boost 后到 1200–1350 MHz（档位 5 / 8）
    上游 constants.yaml 的 bf16 峰值是在 **1350 MHz（档位 8）** 上量的，
    所以"工作档"不能写死成一个数：一律记录本次实测的 level + MHz。
    拿 1.15 GHz（档位 4）算峰值会偏低约 4%。
  - hy-smi 没有 setclocks，但有 --save/--load + --setperflevel/--setsclk；
    --load 会卡在交互式条款提示，必须带 --autorespond Y
  - hy-smi 输出的时钟是**当前 DPM 档位**，不是"额定工作档"：负载会把它推高，
    空闲一段时间又会掉下来。所以 ramp 之后再读，且要连续读几次确认稳定。
"""
from dataclasses import dataclass, field
import contextlib
import fcntl
import os
import re
import subprocess
import tempfile
import time

HY_SMI = "hy-smi"
LOCK_PATH = "/tmp/opbench-gpu.lock"
BUSY_THRESHOLD = 20.0

# 判"这张卡能不能派活"的显存门槛（GB）。
#
# 这个数不是拍脑袋：本仓最大的一道题 `mhc-post-l1` 在 M=16384 档，
# 参考实现峰值 4.91 GB（实测 2026-11-26）。留一倍余量给 agent 自己的
# 临时缓冲和 Triton/tilelang 的 JIT，取 10 GB。
# 改这个值时请同时确认 `python3 -m harness.ready` 的 correctness 项仍通过。
MIN_FREE_VRAM_GB = 10.0

# 实测档位 → MHz（hy-smi -s 的 Supported sclk frequencies）
SCLK_LEVELS_MHZ = {
    0: 300, 1: 600, 2: 800, 3: 1000, 4: 1150,
    5: 1200, 6: 1250, 7: 1300, 8: 1350,
}
# "工作档"判据：不写死 MHz，写死"不低于上游标定峰值时的档位"。
# 上游 bf16 achievable 峰值标注 at_sclk_mhz=1350（档位 8）。
WORK_LEVEL_MIN = 8
IDLE_LEVEL_MAX = 2  # 档位 <=2（<=800 MHz）视为空闲降频

_G_LEVEL = re.compile(r"sclk clock level:\s*(\d+)\s*\((\d+)\s*MHz\)", re.I)
_U_PCT = re.compile(r"HCU\[(\d+)\]\s*:\s*HCU use \(%\):\s*([\d.]+)")
_P_LEVEL = re.compile(r"HCU\[(\d+)\]\s*:\s*Performance Level:\s*(\w+)")
_DEVICE = re.compile(r"HCU\[(\d+)\]")


@dataclass
class DeviceInfo:
    count: int
    name: str
    l2_bytes: int
    sclk_mhz: int | None
    busy_pct: list[float]
    sclk_level: int | None = None
    perf_level: str | None = None
    per_device_mhz: list[int] = field(default_factory=list)


def _run(args: list[str], timeout: float = 15.0) -> str:
    """跑 hy-smi。失败返回空串——探测函数不该因为工具抽风而抛异常。"""
    try:
        p = subprocess.run(
            [HY_SMI, *args], capture_output=True, text=True, timeout=timeout
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    return p.stdout or ""


def _sclk_levels() -> list[tuple[int, int]]:
    """返回 [(level, mhz), ...]，每张卡一条。解析不到返回 []。"""
    out = []
    for m in _G_LEVEL.finditer(_run(["-g"])):
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _busy_pcts() -> list[float]:
    return [float(m.group(2)) for m in _U_PCT.finditer(_run(["-u"]))]


def _perf_levels() -> list[str]:
    return [m.group(2) for m in _P_LEVEL.finditer(_run(["-p"]))]


def query_devices() -> DeviceInfo:
    import torch

    if not torch.cuda.is_available():
        return DeviceInfo(count=0, name="", l2_bytes=0, sclk_mhz=None, busy_pct=[])
    props = torch.cuda.get_device_properties(0)
    levels = _sclk_levels()
    pcts = _busy_pcts()
    perf = _perf_levels()
    mhz = [m for _, m in levels]
    return DeviceInfo(
        count=torch.cuda.device_count(),
        name=props.name,
        l2_bytes=props.L2_cache_size,
        sclk_mhz=min(mhz) if mhz else None,
        busy_pct=pcts,
        sclk_level=min((lv for lv, _ in levels), default=None),
        perf_level=perf[0] if perf else None,
        per_device_mhz=mhz,
    )


def current_sclk() -> int | None:
    """读 hy-smi -g。解析不到就返回 None（不要假装读到）。

    多卡取**最小值**：只要有一张卡在降频，拿别的卡的读数当"机器状态"
    就会把降频状态测成工作档。

    ⚠️ 这是"整机状态"，**不是"我正在用的那张卡"**。绑卡测量请用
    `sclk_of_device(dev)`：实测出现过测的是 1350 的卡 4、却因为别的卡
    在 600 而把基线记成 600 的情况。
    """
    return current_sclk_detail()["min_mhz"]


def sclk_of_device(dev: int | None = None) -> int | None:
    """读**指定卡**的 sclk（MHz）。dev=None 时用 CUDA_VISIBLE_DEVICES 的第一张。

    为什么要单独一个函数：`current_sclk()` 取全卡最小值，是为"整机能不能测"
    设计的。但绑卡测量时那个数会误导——实测卡 4 在 1350、其余在 600，
    基线却记成 600，让人以为是在降频档上测的。
    """
    detail = current_sclk_detail()
    mhz = detail.get("mhz") or []
    if not mhz:
        return None
    if dev is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get(
            "HIP_VISIBLE_DEVICES") or ""
        first = vis.split(",")[0].strip() if vis else ""
        dev = int(first) if first.isdigit() else 0
    if 0 <= dev < len(mhz):
        return mhz[dev]
    return None


def current_sclk_detail(samples: int = 1, interval_s: float = 0.4) -> dict:
    """{levels: [...], mhz: [...], min_mhz, min_level, perf_level}。全解析不到给空。

    samples > 1 时多次采样、按卡取**众数**（并列取低档）。原因：auto DPM
    下，刚跑完一段负载的卡会在 1200/1350 之间抖动一下，单次采样会把
    "本项目自己的探针刚把卡推上去"读成"机器在降频"，每次 doctor 都报一条
    假 warn。众数读的是持续状态。
    """
    snapshots = []
    for i in range(max(1, samples)):
        snapshots.append(_sclk_levels())
        if i + 1 < samples:
            time.sleep(interval_s)

    ncards = max((len(s) for s in snapshots), default=0)
    levels: list[int] = []
    for card in range(ncards):
        seen = [s[card][0] for s in snapshots if len(s) > card]
        if not seen:
            continue
        counts = {lv: seen.count(lv) for lv in set(seen)}
        top = max(counts.values())
        levels.append(min(lv for lv, c in counts.items() if c == top))

    perf = _perf_levels()
    mhz = [SCLK_LEVELS_MHZ.get(lv) for lv in levels]
    mhz = [m for m in mhz if m is not None]
    return {
        "levels": levels,
        "mhz": mhz,
        "min_mhz": min(mhz) if mhz else None,
        "min_level": min(levels) if levels else None,
        "perf_level": perf[0] if perf else None,
    }


def _free_vram_gb(n: int) -> dict[int, float]:
    """每张卡当前可用的 GB。拿不到的卡不出现在结果里（调用方按"未知"处理）。

    只用 `torch.cuda.mem_get_info()`（不初始化 context 也能读，但为稳妥仍
    包在 try 里）：CPU 机器、单元测试、没装 torch 的环境都会走到 except。
    """
    out: dict[int, float] = {}
    try:
        import torch
    except ImportError:
        return out
    for i in range(n):
        try:
            free, _total = torch.cuda.mem_get_info(i)
            out[i] = free / 2**30
        except Exception:
            continue
    return out


def is_busy(threshold: float = BUSY_THRESHOLD) -> bool:
    """**所有**卡都忙才算忙？不——默认仍是"任一卡忙即忙"，见 free_devices。

    保留这个"保守"语义给 doctor 用：体检时要知道整机状态。
    跑 bench 请用 `pick_device()`，它会挑一张空闲卡并绑上去。
    """
    pcts = _busy_pcts()
    if not pcts:
        return True
    return max(pcts) > threshold


def free_devices(threshold: float = BUSY_THRESHOLD,
                 min_free_gb: float = MIN_FREE_VRAM_GB) -> list[int]:
    """返回**既没在算、又够显存**的卡号。

    ## 为什么不能只看占用率

    实测（2026-11-26 凌晨，本机）：8 张卡 HCU 全是 0%，看起来全空闲；
    但卡 0–3 各只剩 4.9–6.9 GB 可用，其余被**容器外**的租户进程占着
    （`hy-smi` 的 VRAM% 是 90%+，而本容器 `ps` 里只有 17 个进程，全是 shell）。

    只看 HCU 会把这几张判成空闲，`--parallel-cards` 随即把题派上去，
    然后第一道 mHC 的 M=16384 档（参考实现峰值 4.9 GB）就 OOM——
    而 OOM 会被记成"这个臂实现有问题"，实际是调度器选错了卡。

    数据源用 `torch.cuda.mem_get_info()` 而不是 hy-smi：前者是本进程
    真正能拿到的量（含显存池碎片），后者是卡级总量，两者不等价。

    读不到显存时**不因此排除**该卡——否则在没有 torch 的环境里
    （单元测试、纯 CPU 机器）会一张卡都返回不了。
    """
    pcts = _busy_pcts()
    if not pcts:
        return []
    mem = _free_vram_gb(len(pcts))
    return [
        i for i, p in enumerate(pcts)
        if p <= threshold and (mem.get(i) is None or mem[i] >= min_free_gb)
    ]


def pick_device(threshold: float = BUSY_THRESHOLD) -> int | None:
    """挑一张空闲卡并**绑定**为当前设备。返回卡号；没有空闲卡返回 None。

    为什么要绑定而不是只用 `torch.cuda.device(i)` 包一层：计时 event、
    torch 的默认流、以及厂商库（aiter / lightop）内部拿的 stream 都跟着
    `current_device` 走。绑一次，后面所有调用都在同一张卡上，避免
    跨设备计时的那个坑（Event 默认绑 cuda:0）。

    为什么要挑卡而不是死等：8 张卡里经常只有一两张被别的会话占着。
    三臂都用**同一张**卡就保持了可比性——卡间差异不会混进 Δ。

    挑**可用显存最多**的那张，不是编号最小的：实测本机卡 0–3 常被
    容器外租户占到只剩 5 GB，而卡 4–7 有 139 GB。按编号取会稳定地
    选到最差的那张。显存读不到时退回"编号最小"，保证老行为不变。
    """
    import torch

    free = free_devices(threshold)
    if not free:
        return None
    mem = _free_vram_gb(max(free) + 1)
    dev = max(free, key=lambda i: (mem.get(i, -1.0), -i))
    torch.cuda.set_device(dev)
    return dev


@contextlib.contextmanager
def gpu_lock(path: str = LOCK_PATH, wait_s: float = 0.0, poll_s: float = 5.0):
    """跨进程 GPU 互斥：flock + 占用检测。

    yield 一个 dict：
      {acquired, waited_s, busy_before, busy_after, holder_pid}
    拿不到锁（wait_s=0 且别人持有）时 acquired=False，调用方必须自己决定
    是等待还是放弃——**不要静默跳过检查**。
    """
    started = time.time()
    info = {"acquired": False, "waited_s": 0.0, "busy_before": None,
            "busy_after": None, "holder_pid": None, "path": path}
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info["acquired"] = True
                break
            except BlockingIOError:
                if time.time() - started >= wait_s:
                    try:
                        info["holder_pid"] = os.pread(fd, 64, 0).decode().strip() or None
                    except OSError:
                        pass
                    break
                time.sleep(poll_s)
        info["waited_s"] = round(time.time() - started, 1)
        if info["acquired"]:
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
            info["busy_before"] = _busy_pcts()
            yield info
            info["busy_after"] = _busy_pcts()
        else:
            yield info
    finally:
        if info["acquired"]:
            try:
                os.ftruncate(fd, 0)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _backup_perf_level(path: str) -> str | None:
    """从 --save 的备份文件里读 perflevel（形如 "perflevel\tauto"）。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "perflevel":
                    return parts[1]
    except OSError:
        pass
    return None


def _save_clocks(path: str) -> bool:
    """hy-smi --save 到 path，成功返回 True。

    坑：**目标文件已存在时 hy-smi 直接放弃**（"already exists. Settings not
    saved"），而退出码仍是 0。所以路径必须是不存在的文件，且保存后要
    验证内容里有 perflevel——否则恢复时读到的是一份空备份，
    看起来"恢复了"其实什么也没做。
    """
    if os.path.exists(path):
        os.unlink(path)
    _run(["--save", path], timeout=20)
    return os.path.isfile(path) and _backup_perf_level(path) is not None


def _restore_from_backup(path: str, want_perf: str | None = None) -> None:
    """恢复时钟与 perf level。

    实测三个坑：
      1. `--save` 拒绝覆盖已存在的文件（见 _save_clocks）
      2. `--load` 会弹 "Do you accept these terms? [Y/N]"，必须 --autorespond Y
      3. **`--load` 对空文件也报 "Successfully loaded"**，所以不能只看它说成功；
         恢复后必须回读状态，不对就显式设回去。
    """
    want = want_perf or _backup_perf_level(path)
    _run(["--autorespond", "Y", "--load", path], timeout=30)
    want = want or "auto"
    if current_sclk_detail()["perf_level"] != want:
        _run(["--setperflevel", want], timeout=20)


def try_lock_clocks(level: int = WORK_LEVEL_MIN) -> bool:
    """试 hy-smi --save / 改 / --load 锁频。失败返回 False，不抛异常。

    实测（2026-11-23）这条路**是通的**，三个坑：
      1. `--save` 拒绝覆盖已存在的文件（且退出码仍是 0）
      2. `--setsclk` 需要先把 perf level 设成 manual，否则静默不生效
      3. `--load` 会卡在 "Do you accept these terms? [Y/N]"，必须 --autorespond Y

    本函数**只验证"改得动"**，不负责把机器锁在工作档。原因是锁频是全机
    共享状态：本会话锁了，另一个会话的 bench 就跟着变。真正的做法是在
    bench 段内 ramp 到工作档并记录实测 MHz（见 ramp_to_work_clock），
    需要锁频的场合由调用方自己 save→setsclk→load 包住整个 bench 段。
    """
    d = tempfile.mkdtemp(prefix="opbench-hysmi-")
    backup = os.path.join(d, "clocks.txt")  # 必须不存在，见 _save_clocks
    before = current_sclk_detail()
    try:
        if not _save_clocks(backup):
            return False
        if before["min_mhz"] is None:
            return False

        # 选一个与当前不同的档位来证明"真的能改"
        target = level if level != before["min_level"] else max(SCLK_LEVELS_MHZ)
        set_ok = bool(
            _run(["--setperflevel", "manual"], timeout=20)
            and _run(["--setsclk", str(target)], timeout=20)
        )
        if not set_ok:
            return False
        changed = current_sclk_detail()["min_mhz"] == SCLK_LEVELS_MHZ.get(target, -1)
        return changed
    except Exception:
        return False
    finally:
        # 无论成败都要还原；不能用 suppress 吞掉还原失败——那是把机器
        # 留在手动档还不吭声。还原失败时至少把 perf level 设回 auto。
        try:
            _restore_from_backup(backup, before["perf_level"])
            if current_sclk_detail()["perf_level"] not in (before["perf_level"], None):
                _run(["--setperflevel", "auto"], timeout=20)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(backup)
            with contextlib.suppress(OSError):
                os.rmdir(d)


def ramp_to_work_clock(timeout_s: int = 30, target_level: int = WORK_LEVEL_MIN) -> int | None:
    """跑 dummy 负载直到 sclk 到工作档，返回最终 sclk。

    "到工作档"= 档位 >= target_level（默认 8，即上游标定峰值时的档位），
    且连续 3 次采样稳定。到不了就返回能达到的最高值——**如实记录**，
    不假装锁到了工作档。
    """
    import torch

    deadline = time.time() + timeout_s
    best = None
    stable = 0
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        while time.time() < deadline:
            for _ in range(40):
                torch.mm(x, y)
            torch.cuda.synchronize()
            info = current_sclk_detail()
            lvl = info["min_level"]
            if lvl is None:
                break
            if best is None or lvl > best:
                best = lvl
                stable = 0
            elif lvl == best:
                stable += 1
            if lvl >= target_level and stable >= 3:
                break
        del x, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    info = current_sclk_detail()
    return info["min_mhz"]


def library_versions() -> dict:
    """lightop / aiter / deepgemm / torch / DTK。不记录就无法复跑。

    importlib.metadata 只看元数据，不 import 包——import lightop 会在
    某些环境下初始化 HCC 运行时，doctor 不该有这种副作用。
    读不到的写 None，不写 "unknown" 冒充一个值。
    """
    from importlib import metadata

    out: dict = {}
    for name in ("lightop", "aiter", "deepgemm", "torch", "triton"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
        except Exception as e:  # pragma: no cover
            out[name] = f"<error: {e}>"

    dtk_dir = "/opt/dtk"
    dtk_ver = None
    for path in (os.path.join(dtk_dir, ".dtk_version"),
                 os.path.join(dtk_dir, ".info", "version-dev")):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    first = f.readline().strip()
                if first:
                    dtk_ver = first
                    break
            except OSError:
                pass
    out["DTK"] = dtk_ver
    out["DTK_path"] = dtk_dir if os.path.isdir(dtk_dir) else None
    return out


def device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def event_timing_works(iters: int = 20) -> tuple[bool, str, float | None]:
    """torch.cuda.Event 能否计时。返回 (ok, 说明, 每次耗时 ms)。

    这是整套测量的地基：异步 kernel 用 time.time() 会被量成 0。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() 为 False", None
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch.mm(a, b)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if not (ms > 0):
            return False, f"elapsed_time 返回 {ms!r}（应 > 0）", ms
        return True, f"{iters} 次 512³ bf16 mm 共 {ms:.3f} ms", ms / iters
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None
