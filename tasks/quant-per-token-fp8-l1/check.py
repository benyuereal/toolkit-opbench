"""题专属机检：量化输出不能逐位比。

## 为什么通用逐位比较不适用

输出是 `q[M,H]` 的 fp8 **码值**。两个都正确的实现（本机 `lightop` 与
`aiter`）在 fp8 尾数**恰好落在舍入中点**的元素上会给出相邻码值，
例如 y=25.0 时一个存 24、一个存 26 —— 逐元素相对误差 8%，
但它们反量化之后都只差 1 个 ULP。

实测（M=2048，随机 bf16 输入）：两者码值不同的元素占 0.0055%，
反量化后的差 rel_l2 = 6.8e-4、max_abs = 0.23（该行 max|x|=5.3）。
拿码值逐位比会把正确实现判错。

## 判据分两部分

### 1. `scale` 必须与定义一致（严格）

`scale = rowmax(|x|) / 448.0`，fp32，形状 `[M, 1]`。这是**量化契约**，
下游 GEMM 用这个 scale 反量化，它错了结果就整体错。所以按 rtol=1e-4 核对
——实测正确实现的最大相对差是 0.0（逐位相同），bf16 存 scale 会差 0.4%，
per-tensor 会差 10~30%，qmax 写成 127 会差 3.5 倍。1e-4 把后三者都挡住。

全零行（amax=0）：约定 scale 不得是 0、NaN 或 Inf。本题 workload 不含
全零行，这条只是防止实现里出现除零产生的垃圾值。

### 2. 反量化后与**原始输入**比（有损，故用相对量）

    d = q.float() * scale                      # 反量化
    rel_l2 = ||d - x||_2 / ||x||_2             ≤ 0.05
    max|d - x| / rowmax(|x|)                   ≤ 0.10

阈值依据（2026-11-25 实测，M∈{2048,4096,8192,12040,13312,16384} ×
{normal, ×1e4, ×1e-4, 交替符号, 全 1}）：

| 实现 | rel_l2 | max|d-x|/rowmax |
|---|---|---|
| aiter.dynamic_per_token_scaled_quant | 0.0265 | 0.0355 |
| lightop.per_token_quant_fp8 | 0.0265 | 0.0355 |
| aiter.per_token_quant_triton | 0.0265 | 0.0355 |
| 截断代替 RNE（仍算合理实现） | 0.0271 | 0.0379 |

rel_l2 取 0.05 = 实测正确值的 1.9 倍余量；逐元素上界取 0.10 = 2.8 倍。
**用行最大值归一是为了尺度无关**：压力档把输入乘 1e4 后，绝对误差到
1755，但那只是尺度变了，误差占比不变（实测仍是 0.033）。用绝对 atol
会在压力档上误判。

## 这条判据抓什么、不抓什么

抓：没做 per-token（scale 全同 → scale 检查挡）、qmax 写错（scale 检查挡）、
scale 存成 bf16（1e-4 挡）、输出没反量化价值（全 0 / 全是 x 的拷贝）、
形状或 dtype 不符、NaN/Inf。

不抓：RNE vs 截断（两者误差只差 0.2 个百分点，属合理实现差异）。
"""
import torch

TOL_SCALE_REL = 1e-4   # 实测正确实现 0.0；bf16 scale 0.4%
REL_L2_MAX = 0.05      # 实测正确实现 0.0265
ELEM_MAX = 0.10        # 实测正确实现 0.0355（已按行最大值归一）


def _as_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(v)


def _compare_pair(a, e) -> tuple[bool, str]:
    """a/e 是 (q[M,H], scale[M,1])，按上面的判据比。"""
    if not isinstance(a, (tuple, list)) or len(a) != 2:
        return False, "输出必须是 (q[M,H], scale[M,1]) 二元组"
    q_a, s_a = _as_tensor(a[0]), _as_tensor(a[1])
    q_e, s_e = _as_tensor(e[0]), _as_tensor(e[1])

    if q_a.shape != q_e.shape:
        return False, f"q 形状不符：{tuple(q_a.shape)} vs {tuple(q_e.shape)}"
    if s_a.shape != s_e.shape:
        return False, f"scale 形状不符：{tuple(s_a.shape)} vs {tuple(s_e.shape)}"
    if q_a.dtype != q_e.dtype:
        return False, f"q dtype 不符：{q_a.dtype} vs {q_e.dtype}"
    if s_a.dtype != s_e.dtype:
        return False, f"scale dtype 不符：{s_a.dtype} vs {s_e.dtype}（约定 fp32）"

    sf_a = s_a.float().reshape(-1)
    sf_e = s_e.float().reshape(-1)
    if not torch.isfinite(sf_a).all():
        n = int((~torch.isfinite(sf_a)).sum())
        return False, f"scale 有 {n} 个 NaN/Inf"
    if not torch.isfinite(q_a.float()).all():
        n = int((~torch.isfinite(q_a.float())).sum())
        return False, f"q 有 {n} 个 NaN/Inf"

    # ---- 1) scale 契约 ----
    nz = sf_e > 0
    if nz.any():
        rel = ((sf_a[nz] - sf_e[nz]).abs() / sf_e[nz]).max()
        rel = float(rel)
        if rel > TOL_SCALE_REL:
            # 给出最像哪种错，便于人读。按误差量级分桶：
            #   bf16 存 scale  → 相对差 ~1e-3 级（10 位尾数）
            #   qmax 写错      → 整体差 3.5×（448/127）或 2×（256/128）
            #   per-tensor/轴错 → 差得很离谱（各行的比值都不一样）
            ratio = float((sf_a[nz] / sf_e[nz]).median())
            hint = ""
            if rel < 5e-3:
                hint = "（scale 只差千分之几 → 可能用 bf16 存了 scale，约定 fp32）"
            elif 3.0 < ratio < 4.0:
                hint = "（scale 约大 3.5× → quant_max 写成了 127 而不是 448）"
            elif 1.8 < ratio < 2.2:
                hint = "（scale 约大 2× → quant_max 写成了 256 或 224？约定 448）"
            else:
                hint = ("（各行 scale 的偏差不一致 → 归约轴错了，或做成了 "
                        "per-tensor / per-channel）")
            return False, f"scale 与定义不一致：最大相对差 {rel:.3e} > {TOL_SCALE_REL:E}{hint}"
    zero_rows = ~nz
    if zero_rows.any():
        bad = sf_a[zero_rows] <= 0
        if bad.any():
            return False, f"{int(bad.sum())} 个全零行的 scale <= 0（应为 1.0 或正数）"

    # ---- 2) 反量化后与原始输入比 ----
    d_a = q_a.float() * s_a.float()
    d_e = q_e.float() * s_e.float()
    if d_a.shape != d_e.shape:
        return False, f"反量化后形状不符：{tuple(d_a.shape)} vs {tuple(d_e.shape)}"

    rowmax = d_e.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    diff = (d_a - d_e).abs()

    rel_l2 = float((d_a - d_e).norm() / d_e.norm().clamp(min=1e-30))
    if rel_l2 > REL_L2_MAX:
        return False, (f"反量化后相对 L2 误差 {rel_l2:.4%} 超过 {REL_L2_MAX:.0%}"
                       f"（该形状 max|d-x|/rowmax = "
                       f"{float((diff / rowmax).max()):.4f}）")

    elem = float((diff / rowmax).max())
    if elem > ELEM_MAX:
        bad = int((diff / rowmax > ELEM_MAX).sum())
        return False, (f"{bad} 个元素的反量化误差超过该行最大值的 {ELEM_MAX:.0%}"
                       f"（最大 {elem:.4f}）；rel_l2={rel_l2:.4%} 本身合规，"
                       "说明是局部元素错，不是整体精度问题")
    return True, ""


def compare_task(actual, expected, tol: dict, defn=None) -> tuple[bool, str]:
    """harness 的题目 hook。actual/expected 均为 (q, scale)。"""
    return _compare_pair(actual, expected)


def check(workdir: str, results: dict) -> list[str]:
    """本题的可跑清单由 harness 主流程负责；这里只做语义级额外校验。"""
    return []
