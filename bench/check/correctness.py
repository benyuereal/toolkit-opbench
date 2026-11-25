"""数值正确性：对参考实现逐 shape 对比，含 holdout 与压力数值。

## 为什么三条都要

* **public**：题面给过的形状。过不了就是彻底错了。
* **holdout**：题面里**没有**的形状。这是防"只对着题面形状调参"的唯一手段——
  SOL-ExecBench 用 seq_len=131/293 就是这个用意。
* **压力数值**：极端输入（大值、小值、非对齐形状）。快但溢出/下溢的实现
  在常规形状上看起来完全正常，只在极端值上崩。这是我们"快但算错"的主防线。

## 容差来源

只从 `definition.tolerance` 读，**不许临时放宽**。放宽容差是"让测试通过"
最省力的做法，也是让整个评测失去意义最快的方法。
"""
from dataclasses import dataclass, field


@dataclass
class Result:
    ok: bool
    max_abs: float | None = None
    cos: float | None = None
    failed_workloads: list[str] | None = None
    note: str = ""


def compare(actual, expected, tol: dict) -> Result:
    """按 rtol/atol 比。tolerance 来自 definition，不许临时放宽。

    判据用 `|a-e| <= atol + rtol*|e|`（与 torch.testing 同口径），
    但额外报 cos 相似度：数值全对但 cos 低说明整体符号/布局错了，
    这种错误用逐元素阈值可能刚好擦边通过。
    """
    import torch

    if not isinstance(tol, dict) or "rtol" not in tol or "atol" not in tol:
        raise ValueError(f"tolerance 必须含 rtol/atol，实际 {tol!r}")
    rtol = float(tol["rtol"])
    atol = float(tol["atol"])

    a = actual.float()
    e = expected.float()
    if a.shape != e.shape:
        return Result(
            ok=False,
            failed_workloads=["shape"],
            note=f"shape 不符：actual {tuple(a.shape)} vs expected {tuple(e.shape)}",
        )
    if not torch.isfinite(a).all():
        bad = int((~torch.isfinite(a)).sum().item())
        return Result(
            ok=False, failed_workloads=["non-finite"],
            note=f"实际输出有 {bad} 个 NaN/Inf（题目没声明允许）",
        )

    diff = (a - e).abs()
    limit = atol + rtol * e.abs()
    ok = bool((diff <= limit).all())
    max_abs = float(diff.max()) if diff.numel() else 0.0

    denom = (a.norm() * e.norm()).clamp(min=1e-12)
    cos = float((a.flatten() @ e.flatten()) / denom) if a.numel() else 0.0

    out = Result(ok=ok, max_abs=max_abs, cos=cos)
    if not ok:
        n_bad = int((diff > limit).sum().item())
        out.failed_workloads = [f"{n_bad} 个元素超差"]
        out.note = (
            f"max_abs={max_abs:.6g} 超过 atol={atol}+rtol={rtol}*|expected|；"
            f"超差元素 {n_bad}/{diff.numel()}，cos={cos:.4f}"
        )
    return out


def run_reference(defn, workload, ref_module) -> object:
    """按 workload 造输入并跑参考实现。

    参考实现约定暴露 `arun(*tensors) -> outputs`，参数顺序与
    definition.inputs 的键序一致。不猜其它调用约定——猜错会静默地
    把输出对到错误的参数上。
    """
    from bench.data import workload as wl_mod

    if not hasattr(ref_module, "arun"):
        raise AttributeError(
            "参考实现必须暴露 arun(*tensors)；没有它就无法重跑核对"
        )
    tensors = wl_mod.build_inputs(workload, defn)
    args = [tensors[name] for name in defn.inputs]
    return ref_module.arun(*args)


_PRESSURE = {
    "large": 1e4,
    "tiny": 1e-4,
    "mixed_sign": 1.0,   # 符号交错：破掉"全正数"实现
}


def build_pressure_inputs(tensors: dict, kind: str = "large") -> dict:
    """把常规输入改成压力数值。只改浮点，整数保持（int8 溢出另有含义）。"""
    import torch

    scale = _PRESSURE.get(kind)
    if scale is None:
        raise ValueError(f"未知压力类型 {kind!r}，可选 {list(_PRESSURE)}")
    out = {}
    for k, v in tensors.items():
        if not torch.is_floating_point(v):
            out[k] = v
            continue
        if kind == "mixed_sign":
            flat = v.flatten()
            signs = torch.ones_like(flat)
            signs[1::2] = -1
            out[k] = (flat.abs() * signs).reshape(v.shape)
        else:
            out[k] = v * scale
    return out


_HOOK_CACHE: dict[str, object] = {}


def _task_hook(defn):
    """加载题目目录下的 check.py:compare_task（如果定义了）。

    为什么需要：通用比法是逐位 + 浮点容差。有些算子的输出**顺序不唯一**
    （moe-align 的 sorted_ids：同一专家内 token 先后由实现决定），
    或有「只有前缀有效」的约定（尾部是缓冲区残留）。逐位比会把这些
    正确实现判错，而放宽容差又会放过真错。所以让题目自己声明怎么比。

    题目的 check.py 里没有 compare_task 就返回 None，走通用路径。
    结果按 task id 缓存：一个 check_all 会对同一题跑十来个 workload
    （public + holdout + 三档压力），不缓存就是反复 exec_module。
    """
    import importlib.util
    import os

    tid = getattr(defn, "id", None)
    if not tid:
        return None
    if tid in _HOOK_CACHE:
        return _HOOK_CACHE[tid]
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "tasks", tid, "check.py")
    hook = None
    if os.path.isfile(path):
        try:
            spec = importlib.util.spec_from_file_location(f"_taskchk_{tid}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            hook = getattr(mod, "compare_task", None)
        except Exception:
            hook = None
    _HOOK_CACHE[tid] = hook
    return hook


def check_all(defn, workloads, solution_fn, ref_module) -> Result:
    """public + holdout + 压力数值（极端 dtype 取值，如 fp16 上溢）。

    `solution_fn` 接受 build_inputs 造的那组张量、返回输出张量（或
    与 outputs 键对应的 dict）。参考实现在同样的输入上跑一遍，逐 shape 比较。
    """
    from bench.data import workload as wl_mod

    tol = defn.tolerance
    if not tol:
        return Result(ok=False, note=f"{defn.id} 没有 tolerance，无法判正确性")

    pub, hold = wl_mod.split(workloads)
    failed: list[str] = []
    max_abs = 0.0
    worst_cos = 1.0
    notes: list[str] = []

    def _run_one(w, pressure: str | None = None):
        nonlocal max_abs, worst_cos
        tensors = wl_mod.build_inputs(w, defn)
        if pressure:
            tensors = build_pressure_inputs(tensors, pressure)
        expected = ref_module.arun(*[tensors[n] for n in defn.inputs])
        actual = solution_fn(tensors)
        if isinstance(actual, dict):
            # 多输出的题：按 outputs 键序取
            actual = [actual[n] for n in defn.outputs]
            if len(actual) == 1:
                actual = actual[0]
        if isinstance(actual, (list, tuple)):
            exps = expected if isinstance(expected, (list, tuple)) else [expected]
            if len(actual) != len(exps):
                return Result(
                    ok=False, failed_workloads=["output-count"],
                    note=f"输出个数不符：actual {len(actual)} vs expected {len(exps)}",
                )
            # 题目可以自己定义比对方式：有些算子的输出顺序不唯一
            # （moe-align 的 sorted_ids 就是），或有"只有前缀有效"的约定。
            # 逐位比会把正确实现判错。题级 hook 优先。
            hook = _task_hook(defn)
            if hook is not None:
                ok, note = hook(actual, expected, tol, defn)
                if not ok:
                    return Result(ok=False, failed_workloads=["task-hook"], note=note)
                return Result(ok=True)
            res = Result(ok=True)
            for idx, (a, e) in enumerate(zip(actual, exps)):
                # 整数输出（top-k 索引、token id、flags）**必须精确相等**。
                # 套浮点容差是错的：索引差 1 是完全不同的元素，而
                # atol=0.01+rtol 会把 54 个"差 1"的索引判成超差 —— 反过来
                # 若 atol 调大又会放过大错。这里按 dtype 分流。
                is_int = (not isinstance(a, float)) and hasattr(a, "dtype") and \
                         (not a.dtype.is_floating_point)
                if is_int:
                    e_int = e.to(a.dtype) if hasattr(e, "to") else e
                    if a.shape != e_int.shape:
                        return Result(ok=False, failed_workloads=["shape"],
                                      note=f"输出[{idx}] shape 不符："
                                           f"{tuple(a.shape)} vs {tuple(e_int.shape)}")
                    n_bad = int((a != e_int).sum().item())
                    if n_bad:
                        return Result(
                            ok=False, failed_workloads=[f"整数输出[{idx}] {n_bad} 个不等"],
                            note=f"整数输出[{idx}] 有 {n_bad}/{a.numel()} 个元素不相等"
                                 "（索引/ID 必须精确匹配，不适用浮点容差）",
                        )
                    continue
                r = compare(a, e, tol)
                if not r.ok:
                    return r
                max_abs = max(max_abs, r.max_abs or 0.0)
                worst_cos = min(worst_cos, r.cos or 0.0)
            return res
        r = compare(actual, expected, tol)
        max_abs = max(max_abs, r.max_abs or 0.0)
        worst_cos = min(worst_cos, r.cos if r.cos is not None else 0.0)
        return r

    for w in pub:
        r = _run_one(w)
        if not r.ok:
            failed.append(f"public:{w.uuid}")
            notes.append(r.note)
    for w in hold:
        r = _run_one(w)
        if not r.ok:
            failed.append(f"holdout:{w.uuid}")
            notes.append(r.note)
    # 压力数值只在 public 上跑：holdout 已覆盖"未提交形状"，
    # 两者叠加会让跑一次的成本翻倍，而信息量不增加。
    #
    # 但**有整数输出的题要跳过压力档**。实测教训：moe-gating 把输入 ×1e4 后，
    # softmax 直接饱和成 one-hot，第 k 名之后全是 0、完全并列，top-k 索引
    # 谁先谁后都对。这时各实现给出的索引集合相同但顺序不同，
    # 拿"索引必须精确相等"去判会把正确的实现判成失败。
    # 整数输出（索引/ID）在饱和输入下本来就没有唯一解，测它没有意义。
    has_int_output = any(
        (not spec.get("dtype", "").startswith("float"))
        for spec in (defn.outputs or {}).values()
    )
    if not has_int_output:
        for w in pub:
            for kind in _PRESSURE:
                r = _run_one(w, pressure=kind)
                if not r.ok:
                    failed.append(f"pressure[{kind}]:{w.uuid}")
                    notes.append(r.note)

    return Result(
        ok=not failed,
        max_abs=max_abs,
        cos=worst_cos if worst_cos < 1.0 else 1.0,
        failed_workloads=failed or None,
        note="；".join(dict.fromkeys(n for n in notes if n))[:600],
    )
