"""workload.jsonl：形状实例与输入生成。

一行一个实例：axes 取值 + inputs 生成方式 + split(public/holdout)。
holdout 只在评分时用，不能出现在题面里。

为什么输入生成必须由 uuid 派生 seed：同一个 workload 在三次评测里
必须是同一组数，否则「快 3%」可能只是换了一组随机数。用 Python 的
hash() 不行——它每个进程不同（PYTHONHASHSEED）。
"""
from dataclasses import dataclass
import hashlib
import json

SPLITS = ("public", "holdout")
INPUT_TYPES = ("random", "zeros", "ones", "scalar", "arange")


@dataclass
class Workload:
    uuid: str
    axes: dict
    inputs: dict
    split: str = "public"

    def is_holdout(self) -> bool:
        return self.split == "holdout"


def _validate_instance(d: dict, lineno: int, axis_names: set[str], var_names: set[str]) -> list[str]:
    where = f"workload.jsonl:{lineno}"
    problems: list[str] = []
    uuid = d.get("uuid")
    if not isinstance(uuid, str) or not uuid.strip():
        problems.append(f"{where}: uuid 必须是非空字符串")
    split = d.get("split", "public")
    if split not in SPLITS:
        problems.append(f"{where}: split 必须是 {SPLITS}，实际 {split!r}")
    axes = d.get("axes")
    if not isinstance(axes, dict):
        problems.append(f"{where}: axes 必须是对象")
        axes = {}
    for name, val in axes.items():
        if name not in axis_names:
            problems.append(f"{where}: axes 出现未声明的轴 {name!r}")
        elif name in var_names:
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                problems.append(f"{where}: axes.{name} 必须是正整数，实际 {val!r}")
        else:
            problems.append(f"{where}: axes.{name} 是 const 轴，不该在 workload 里给值")
    missing = var_names - set(axes)
    if missing:
        problems.append(f"{where}: 缺少 var 轴取值 {sorted(missing)}")
    inputs = d.get("inputs", {})
    if not isinstance(inputs, dict):
        problems.append(f"{where}: inputs 必须是对象")
    else:
        for name, spec in inputs.items():
            if not isinstance(spec, dict):
                problems.append(f"{where}: inputs.{name} 必须是对象")
                continue
            t = spec.get("type")
            if t not in INPUT_TYPES:
                problems.append(f"{where}: inputs.{name}.type {t!r} 不在 {INPUT_TYPES}")
            if t == "scalar" and "value" not in spec:
                problems.append(f"{where}: inputs.{name} 是 scalar，必须给 value")
    return problems


def load(path: str, definition=None) -> list[Workload]:
    """读 workload.jsonl。给了 definition 就顺带校验轴名与 split。"""
    ws: list[Workload] = []
    problems: list[str] = []
    axis_names: set[str] = set()
    var_names: set[str] = set()
    if definition is not None:
        axis_names = set(definition.axes)
        var_names = set(definition.var_axes())

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{e}") from e
            if not isinstance(d, dict):
                raise ValueError(f"{path}:{lineno} 必须是 JSON 对象")
            if definition is not None:
                problems.extend(_validate_instance(d, lineno, axis_names, var_names))
            ws.append(
                Workload(
                    uuid=d.get("uuid", f"line{lineno}"),
                    axes=d.get("axes", {}),
                    inputs=d.get("inputs", {}),
                    split=d.get("split", "public"),
                )
            )

    if not ws:
        raise ValueError(f"{path} 是空的：至少需要一条 public 与一条 holdout")
    if definition is not None and problems:
        raise ValueError(
            f"{path} 校验失败（{len(problems)} 项）：\n  - " + "\n  - ".join(problems)
        )
    return ws


def split(ws: list[Workload]) -> tuple[list[Workload], list[Workload]]:
    """分成 (public, holdout)。至少各一，否则抛错。

    单边为空是题面构造错误，不是「这次没有留出集」——静默放行会让
    「只在题面形状上调参」的实现拿到满分。
    """
    pub = [w for w in ws if not w.is_holdout()]
    hold = [w for w in ws if w.is_holdout()]
    if not pub:
        raise ValueError("workload 没有 public 实例")
    if not hold:
        raise ValueError("workload 没有 holdout 实例：留出集是防过拟合的唯一手段")
    return pub, hold


def _seed_of(uuid: str) -> int:
    """uuid → 稳定 63bit seed（与进程、平台无关）。"""
    digest = hashlib.sha256(uuid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _dtype(name: str):
    import torch

    dt = getattr(torch, name, None)
    if dt is None:
        raise ValueError(f"torch 没有 dtype {name!r}")
    return dt


def build_inputs(w: Workload, d) -> dict:
    """按 inputs 声明造张量。要支持 type=random / zeros / scalar。
    生成必须可复现（固定 seed 由 uuid 派生）。

    ## 整数张量的取值范围必须由题面声明

    默认 `randint(0, 8)` 对"索引"类输入是错的：`mla-sparse-l1` 的
    `indices` 必须是 [0, SKV) 范围，而且哨兵 `-1` 要按比例出现。
    实测踩过：默认生成只产出 0–7，SKV=8192，且一个 -1 都没有——
    哨兵语义（本题三大考点之一）根本不被触发。

    所以在 definition.inputs.<name> 里可以声明：

      "low": 0,               # 下界（含）
      "high": "SKV",          # 上界（不含），可以是轴名
      "invalid_ratio": 0.3,   # 额外按此比例填入 "invalid_value"
      "invalid_value": -1
    """
    import torch

    axes = dict(d.const_axes())
    axes.update(w.axes)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev)
    gen.manual_seed(_seed_of(w.uuid))

    def _resolve(v, default):
        if v is None:
            return default
        if isinstance(v, str):
            if v not in axes:
                raise ValueError(f"inputs 的边界引用了未声明的轴 {v!r}")
            return axes[v]
        return int(v)

    tensors: dict = {}
    for name, spec in (d.inputs or {}).items():
        shape = tuple(axes[dim] for dim in spec["shape"])
        dt = _dtype(spec["dtype"])
        decl = (w.inputs or {}).get(name, {"type": "random"})
        t = decl.get("type", "random")

        if t == "random":
            if dt.is_floating_point:
                # 量化 scale 这类输入必须是**正的小数**，randn 出的量级（±3）
                # 完全不对：`mla-sparse-glm53-fp8-l1` 的 kscale 实测在 1e-2 量级。
                # 用 randn*0.03 会让反量化后的 kv 比真实值大两个数量级，
                # 而正确性判据（相对误差）对这个不敏感——于是题目"能过"，
                # 但测的是一个不存在的数值范围。声明 low_scale/high_scale
                # 就在这个区间里均匀取，与真实分布同量级。
                ls, hs = spec.get("low_scale"), spec.get("high_scale")
                if ls is not None or hs is not None:
                    lo_f = float(ls if ls is not None else 0.0)
                    hi_f = float(hs if hs is not None else 1.0)
                    if hi_f <= lo_f:
                        raise ValueError(
                            f"inputs.{name}: high_scale({hi_f}) 必须大于 "
                            f"low_scale({lo_f})")
                    x = torch.rand(shape, dtype=torch.float32, device=dev,
                                   generator=gen) * (hi_f - lo_f) + lo_f
                    x = x.to(dt)
                else:
                    # fp8 没有 randn 实现（`normal_kernel_cuda not implemented
                    # for Float8_e4m3fn`），先生成 bf16 再 cast。
                    # 直接对 fp8 调 randn 是实测踩过的错。
                    _dt_gen = torch.bfloat16 if dt in (
                        torch.float8_e4m3fn, torch.float8_e5m2) else dt
                    x = torch.randn(shape, dtype=_dt_gen, device=dev,
                                    generator=gen).to(dt)
                    # 某些输入在真实模型里**不可能是负的**，拿 randn 造会造出
                    # 物理上不存在的算子：`mhc-post-l1` 的 `comb` 是 Sinkhorn 的
                    # 输出（非负、行和≈1），`post` 过 sigmoid（∈[0,2]）。负的
                    # 混合权重会让 4 项互相抵消到 1e-8，正确性判据就变成
                    # "谁的舍入误差更小"。声明 `positive: true` 取 |randn|。
                    if decl.get("positive"):
                        x = x.abs()
            else:
                lo = _resolve(spec.get("low"), 0)
                hi = _resolve(spec.get("high"), 8)
                if hi <= lo:
                    raise ValueError(
                        f"inputs.{name}: high({hi}) 必须大于 low({lo})")
                x = torch.randint(lo, hi, shape, dtype=dt, device=dev, generator=gen)
                ratio = spec.get("invalid_ratio")
                if ratio:
                    val = spec.get("invalid_value", -1)
                    mask = torch.rand(shape, device=dev, generator=gen) < float(ratio)
                    x = torch.where(mask, torch.full_like(x, int(val)), x)
        elif t == "zeros":
            x = torch.zeros(shape, dtype=dt, device=dev)
        elif t == "ones":
            x = torch.ones(shape, dtype=dt, device=dev)
        elif t == "arange":
            x = torch.arange(int(torch.tensor(shape).prod()), dtype=dt, device=dev)
            x = x.reshape(shape)
        elif t == "scalar":
            x = torch.full(shape, decl["value"], dtype=dt, device=dev)
        else:
            raise ValueError(f"inputs.{name}.type {t!r} 不受支持")
        tensors[name] = x
    return tensors


def generate_holdout(d, n: int = 3, seed: int = 0) -> list[Workload]:
    """从 var 轴的取值域里采未出现过的形状。用于补 holdout。

    只支持单 var 轴或全部 var 轴共用一个 domain 的简单情形；域里给出
    values 就按 values 采，给出 min/max 就在区间内按固定 seed 采。
    不猜轴之间的相关性（例如 M 与 N 联动），宁可少采也不编。
    """
    import random

    rng = random.Random(seed)
    var = d.var_axes()
    if not var:
        raise ValueError(f"{d.id} 没有 var 轴，无法生成 holdout")

    domains = {}
    for name in var:
        spec = d.axes[name]
        dom = spec.get("domain") or {}
        domains[name] = dom

    out: list[Workload] = []
    used: set[tuple] = set()
    for i in range(n * 20):
        if len(out) >= n:
            break
        vals = {}
        for name in var:
            dom = domains[name]
            if "values" in dom:
                vals[name] = rng.choice(dom["values"])
            else:
                lo = int(dom.get("min", 1))
                hi = int(dom.get("max", lo))
                vals[name] = rng.randint(lo, hi)
        key = tuple(sorted(vals.items()))
        if key in used:
            continue
        used.add(key)
        out.append(
            Workload(
                uuid=f"gen-holdout-{seed}-{len(out)}",
                axes=vals,
                inputs={k: {"type": "random"} for k in (d.inputs or {})},
                split="holdout",
            )
        )
    if len(out) < n:
        raise ValueError(
            f"{d.id} 的 var 轴取值域组合不足：要 {n} 个 holdout，只采到 {len(out)} 个"
        )
    return out
