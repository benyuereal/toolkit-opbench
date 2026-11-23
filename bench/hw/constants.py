"""硬件常数的唯一来源。

**禁止在代码里写 560 / 310 这类字面量。** 一律按 id 从上游 constants.yaml 读，
原因是分母已经错过两次（540 无来源、49.8 当峰值），错一次结论全反。

## id 语法

只认三种，对应上游的 `caliber`：

    peak_nominal:<precision>     硬件名义峰值，不可复跑，只能当「上限」
    peak_achievable:<precision>  本机某高质量实现实测，可复跑，当「还有多少空间」的分母
    impl_measured:<precision>    某一条具体实现的值，只代表那条实现

也接受上游的点分路径（`compute.bf16_gemm_peak_tf`），kind 由节点自带的
`caliber` 推出；节点没有 caliber 的（时钟、L2 这类）返回 kind=None。

## 为什么要这么严

同一个 49.8，被当成 impl_measured 时说明「这条实现慢」，被当成
achievable/上限时说明「几乎没空间」——结论完全相反。所以取值时**必须**
带上类别，`peak()` 也强制要求 kind。

## 2026-11-23 追加：时钟档位

上游 `compute.clock_ghz = 1.15` 记的是「生产卡 6-9 的档位」，但本机实测
ramp 之后 hy-smi 读到的稳定档位是 **level 5 = 1200 MHz**（空闲 600 MHz）。
用例：拿 1.15 GHz 算峰值，会比按 1.20 GHz 算**偏低约 4%**；反过来若把
1.15 当成「工作档」去判「现在降频了」，会误判 1200 MHz 也是异常。
**工作档一律以 sclk 实测为准**，constants 里的 1.15 只作为「历史上记录过的
档位」保留，不作为「应有档位」。
"""
from functools import lru_cache
import os

UPSTREAM = (
    "/workspace/toolkit_skills/Plugins/ops/skills/op-roofline/constants.yaml"
)
# 只允许引用这三类，不许混
KINDS = ("peak_nominal", "peak_achievable", "impl_measured")

VARIANT = {
    # id -> 上游点分路径。**只有路径，没有数字。**
    ("peak_nominal", "fp8"): "compute.nominal_peak_tf.fp8_nominal_tf",
    ("peak_nominal", "bf16"): "compute.nominal_peak_tf.bf16_nominal_tf",
    ("peak_achievable", "bf16"): "compute.bf16_gemm_peak_tf",
    ("impl_measured", "fp8"): "compute.fp8_impl_measured_tf",
}

# 带宽口径：与算力三类**同一套语义**，但对象是 GB/s。
#
# 为什么要单列：count 型算力的分母与簿记型算力的分母不是一回事。
# 上游的 memory.* 节点没有 caliber 字段，所以 `KINDS` 覆盖不到它们；
# 但语义上 `stream_copy_peak` 就是「本机某高质量实现的实测带宽」，
# 正对应 peak_achievable。`spec_bandwidth` 是型号标称，对应 peak_nominal。
#
# 用错口径的后果与算力那边一样严重：constants 的 conflicts 段记着，
# 同一个 955 GB/s 拿 1100 当分母是 87%（"到头了"），
# 拿 1724 当分母是 55%（"还有 1.8×"）。分母换了，结论就反了。
BANDWIDTH = {
    ("peak_achievable", "copy"): "memory.stream_copy_peak",
    ("peak_achievable", "read"): "memory.stream_read_peak",
    ("peak_nominal", "ddr"): "memory.spec_bandwidth",
}

# 时钟档位的实测注记（本仓库实测，2026-11-23；数字是观测值，不是性能分母）
SCLK_NOTE = {
    "idle_mhz_observed": 600,
    "work_level_observed": 5,
    "work_mhz_observed": 1200,
    "upstream_clock_ghz_note": "上游记 1.15 GHz；按 1.2 GHz 折算峰值会差约 4%",
    "rule": "工作档以 hy-smi 实测 sclk 为准，不以上游 clock_ghz 反推",
}


class ConstantsError(RuntimeError):
    """上游常数文件缺失或读不了。不 fallback 到默认值。"""


@lru_cache(maxsize=1)
def load_all() -> dict:
    path = UPSTREAM
    if not os.path.isfile(path):
        raise ConstantsError(
            f"找不到常数文件 {path}。它是峰值的唯一来源，缺失时不许编一个默认值；"
            "请挂载 toolkit_skills 或改 constants.UPSTREAM。"
        )
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ConstantsError(f"读常数需要 PyYAML：{e}") from e
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConstantsError(f"{path} 顶层不是映射")
    return data


def reload() -> None:
    """丢掉缓存（换机器 / 上游改了之后用）。"""
    load_all.cache_clear()


def _walk(node, prefix: str, out: dict) -> None:
    if isinstance(node, dict):
        # 带 value 的节点本身就是常量，登记它的路径；同时继续往下走，
        # 因为有的节点既有 value 又有子常数（如 nominal_peak_tf）
        if "value" in node:
            out[prefix] = node
        for k, v in node.items():
            _walk(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = node


def _index() -> dict:
    out: dict = {}
    _walk(load_all(), "", out)
    return out


def _kind_of(node: dict) -> str | None:
    caliber = node.get("caliber")
    if caliber is None:
        return None
    mapping = {
        "nominal": "peak_nominal",
        "achievable": "peak_achievable",
        "impl_measured": "impl_measured",
    }
    if caliber not in mapping:
        raise ConstantsError(f"上游出现未知 caliber={caliber!r}，拒绝猜它属于哪一类")
    return mapping[caliber]


def _node(path: str) -> dict:
    node = load_all()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _result(path: str, kind: str | None = None) -> dict:
    node = _node(path)
    if not isinstance(node, dict) or "value" not in node:
        # 上游有些叶子直接是数字（如 compute.nominal_peak_tf.fp8_nominal_tf），
        # 元数据（unit/status/caliber）挂在父节点上。取父节点元数据，值取叶子。
        parent_path, _, _ = path.rpartition(".")
        try:
            parent = _node(parent_path) if parent_path else None
        except KeyError:
            parent = None
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            value = node
            meta = parent if isinstance(parent, dict) else {}
            node = dict(meta)
            node["value"] = value
        else:
            raise KeyError(path)
    declared = _kind_of(node)
    if kind is not None and declared is not None and declared != kind:
        raise ConstantsError(
            f"{path} 的 caliber 是 {declared}，被按 {kind} 引用；类别不许混"
        )
    return {
        "id": path,
        "value": node["value"],
        "unit": node.get("unit"),
        "kind": declared if kind is None else kind,
        "status": node.get("status"),
        "source": node.get("source"),
        "at_sclk_mhz": node.get("at_sclk_mhz"),
    }


def get(const_id: str) -> dict:
    """返回 {value, unit, kind, status, source}。查不到抛 KeyError，
    不许 fallback 到默认值——那等于又编了一个数。"""
    if not isinstance(const_id, str) or not const_id.strip():
        raise KeyError("常数 id 必须是非空字符串")
    cid = const_id.strip()

    if ":" in cid:
        kind, _, precision = cid.partition(":")
        kind = kind.strip()
        precision = precision.strip()
        if kind not in KINDS:
            raise KeyError(f"{cid}: 类别必须是 {KINDS}")
        # 先在算力表查，再在带宽表查 —— 两张表共用 `类别:对象` 的语法，
        # 但对象不同（fp8/bf16 是精度，copy/read/ddr 是访存模式）。
        path = VARIANT.get((kind, precision))
        if path is None:
            path = BANDWIDTH.get((kind, precision))
        if path is None:
            have = sorted(f"{k}:{p}" for k, p in VARIANT) + \
                   sorted(f"{k}:{p}" for k, p in BANDWIDTH)
            raise KeyError(
                f"{cid}: 上游没有这个 {kind}/{precision} 值。现有：{have}。"
                "缺就如实报缺，不要拿别的精度/模式顶替"
                "（fp8/bf16 是 2:1 双速率；read/copy 是两个不同的带宽口径）"
            )
        return _result(path, kind)

    # 点分路径：先看是不是完整叶子
    try:
        return _result(cid)
    except KeyError:
        pass
    # 再看唯一后缀匹配（如 bf16_gemm_peak_tf）
    matches = [k for k in _index() if k.endswith("." + cid)]
    if len(matches) == 1:
        return _result(matches[0])
    if len(matches) > 1:
        raise KeyError(f"{cid}: 命中多条 {matches}，请写完整点分路径")
    raise KeyError(
        f"{cid}: constants.yaml 里没有这个 id。"
        "只许引用上游 id，不许在代码里写字面量"
    )


def peak(kind: str, precision: str) -> float:
    """按 类别+精度 取峰值。kind 必须是 KINDS 之一。

    对带宽类（`precision` 取 copy/read/ddr），结果单位是 GB/s；
    对算力类（fp8/bf16），单位是 TFLOPS。调用方必须知道自己在要哪个——
    `peak` 只管 id 解析，不替你判断"这道题该用算力还是带宽"。
    """
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一，实际 {kind!r}")
    return float(get(f"{kind}:{precision}")["value"])


def bandwidth(kind: str, mode: str) -> float:
    """取带宽峰值（GB/s）。kind ∈ KINDS，mode ∈ BANDWIDTH 的第二键。"""
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一，实际 {kind!r}")
    try:
        path = BANDWIDTH[(kind, mode)]
    except KeyError:
        have = sorted(f"{k}:{m}" for k, m in BANDWIDTH)
        raise KeyError(
            f"{kind}:{mode} 没有带宽常数。现有：{have}。"
            "带宽的分母与算力的分母同样不能编"
        ) from None
    node = _result(path, kind)
    unit = (node.get("unit") or "").upper()
    if unit != "GB/S":
        raise ConstantsError(f"{path} 单位是 {node.get('unit')!r}，不是 GB/s")
    return float(node["value"])


def l2_bytes() -> int:
    """L2 容量（字节）。上游单位 MB。"""
    node = _result("memory.l2_size")
    if (node.get("unit") or "").upper() != "MB":
        raise ConstantsError(f"memory.l2_size 单位是 {node.get('unit')!r}，不是 MB")
    return int(float(node["value"]) * 1024 * 1024)


def upstream_clock_mhz() -> int:
    """上游记录的时钟档位（见模块 docstring 的注记）。不作为「应有档位」。"""
    return int(round(float(_result("compute.clock_ghz")["value"]) * 1000))


def summary() -> dict:
    """doctor 用：三类峰值各有哪些 precision、值是多少、状态是否可引用。"""
    out: dict = {k: {} for k in KINDS}
    for (kind, precision), path in VARIANT.items():
        node = _result(path, kind)
        out[kind][precision] = {
            "value": node["value"],
            "unit": node["unit"],
            "status": node["status"],
        }
    return out
