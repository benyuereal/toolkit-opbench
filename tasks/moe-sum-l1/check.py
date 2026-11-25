"""题专属机检：这道题不能用逐元素相对误差。

## 为什么

bf16 累加 K=8 份，会出现**抵消**：8 个值的和恰好接近 0。
这时绝对误差很小（实测 max 0.154），但相对误差爆表（实测 3515625×）。
逐元素的 `|a-e| <= atol + rtol*|e|` 在这种元素上永远不可能过——
不是实现错了，是判据不适配。

## 用什么代替

**相对 L2 + 逐元素绝对上界**，这是数值线性代数的通行做法：

    rel_l2 = ||a - e||_2 / ||e||_2        ≤ 0.01
    max|a - e|                            ≤ 0.25     （bf16 在量级 8 的 ULP 的 ~8 倍）

实测 `aiter.moe_sum`（正确实现）：rel_l2 = 0.368%、cos = 0.999993、
max_abs = 0.154。两条都过，留了一倍余量。

## 为什么还要留逐元素上界

只有 rel_l2 会给"整体对、局部离谱"放行：把 1% 的元素改错 100 倍，
L2 范数可能只涨几个百分点。加一条绝对值上界堵住这个口子。
"""
import torch

REL_L2_MAX = 0.01      # 1%：实测正确实现 0.368%，留 2.7 倍余量
MAX_ABS_MAX = 0.25     # bf16 量级 8 时 ULP≈0.031，实测正确实现 0.154


def compare_task(actual, expected, tol: dict, defn=None) -> tuple[bool, str]:
    """返回 (ok, note)。"""
    a = actual.float() if hasattr(actual, "float") else torch.as_tensor(actual).float()
    e = expected.float() if hasattr(expected, "float") else torch.as_tensor(expected).float()

    if a.shape != e.shape:
        return False, f"shape 不符：{tuple(a.shape)} vs {tuple(e.shape)}"
    if not torch.isfinite(a).all():
        return False, f"输出有 {int((~torch.isfinite(a)).sum())} 个 NaN/Inf"

    diff = (a - e).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denom = float(e.norm()) or 1e-12
    rel_l2 = float((a - e).norm()) / denom

    if rel_l2 > REL_L2_MAX:
        return False, (f"相对 L2 误差 {rel_l2:.4%} 超过 {REL_L2_MAX:.0%}"
                       f"（max_abs={max_abs:.4f}）")
    if max_abs > MAX_ABS_MAX:
        bad = int((diff > MAX_ABS_MAX).sum())
        return False, (f"有 {bad} 个元素绝对误差超过 {MAX_ABS_MAX}"
                       f"（最大 {max_abs:.4f}）；rel_l2={rel_l2:.4%} 本身合规，"
                       "但局部错得太多，不是精度问题是实现问题")

    return True, ""


def check(workdir: str, results: dict) -> list[str]:
    """本题的可跑清单由 harness 主流程负责；这里只做语义级额外校验。"""
    return []
