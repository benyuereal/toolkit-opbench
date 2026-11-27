"""题专属机检：FP8 输入下不能用逐元素相对误差。

## 为什么

FP8 e4m3 只有 3 位尾数（相对分辨率约 6%），再加上 softmax 之后的加权求和
会出现**抵消**：某些元素的和接近 0，绝对误差很小但相对误差爆表。
逐元素的 `|a-e| <= atol + rtol*|e|` 在这些元素上永远过不了——
不是实现错了，是判据不适配。这与 `moe-sum-l1` 踩的是同一个坑。

## 用什么代替

**相对 L2 + 逐元素绝对上界**：

    rel_l2 = ||a - e||_2 / ||e||_2        ≤ 0.02
    max|a - e|                            ≤ 0.05

阈值依据（实测，S=128，30% 哨兵）：
  * 正确实现（参考实现自己对自己）：rel_l2 = 0、max_abs = 0
  * 反量化 scale 用错粒度（整 token 一个 scale）：rel_l2 ≈ 0.3、max_abs ≈ 2.9
  * 把 -1 当索引 0：rel_l2 ≈ 0.5、max_abs ≈ 2.9

也就是说正确实现与「有 bug 的实现」之间差 15 倍以上，阈值 0.02 留了足够的
区分度，同时容许 FP8 本身的量化噪声（bf16 输出下约 0.5%）。

## 为什么还要留逐元素上界

只有 rel_l2 会给"整体对、局部离谱"放行：把 1% 的元素改错 100 倍，
L2 范数可能只涨几个百分点。加一条绝对值上界堵住这个口子。
"""
import torch

REL_L2_MAX = 0.02      # 2%：正确实现 0%，有 bug 的实现 30%+，中间留 15 倍
MAX_ABS_MAX = 0.05     # 输出量级 ~0.3，FP8 量化噪声 ~0.005，留 10 倍余量


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
