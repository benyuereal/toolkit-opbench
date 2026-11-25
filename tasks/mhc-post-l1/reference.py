"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

GLM-5.3-Flash 的 mHC（hyper-connection）后处理。上一层留下 hc=4 份残差，
这一层把它们按 `comb` 混合回 4 份，再加上本层输出的 4 个缩放副本：

    out[m, i, h] = post[m, i] * x[m, h]
                 + Σ_{j=0}^{3} comb[m, i*4 + j] * residual[m, j, h]

拆开看是两件事：

  ① 广播缩放：`x[m, H]` 按 4 个不同的标量各缩一次 → `[m, 4, H]`。
     这一项每个 i 都要重新乘一遍同一份 x，**不动 residual 的对角线**。
  ② 4×4 混合：`residual[m, 4, H]` 沿残差轴做一个逐元素位置的 4×4 矩阵乘。
     H 是 4096，矩阵只有 4 行——这是**内存受限**的，不是 GEMM。

## 流量口径（definition.dcu.work）

    x         [M, H]      bf16   读   M*H*2        = 1H
    residual  [M, 4, H]   bf16   读   M*4*H*2      = 4H
    out       [M, 4, H]   bf16   写   M*4*H*2      = 4H
    post      [M, 4]      fp32   读   M*4*4        ≈ 0
    comb      [M, 16]     fp32   读   M*16*4       ≈ 0
    ----------------------------------------------------
    合计 9H（每 token 9*4096*2 = 73728 字节）

`post`/`comb` 虽然逐 token 都读，但总量是 M*80 字节，相对 M*73728 可忽略
（0.1%）。不把它们算进"可优化空间"是**有意为之**：它们的成本是访存
粒度问题，不是带宽问题。

## 为什么这题值得出

`out` 里的 4 个 i 各需要读一遍完整的 `residual[j, :]`（4 份）和 `x`（1 份）。
如果按 i 分块，`residual` 会被读 4 遍 —— 流量从 9H 变成 1H+16H+4H = 21H。
**正确做法是让一个 CTA 同时持有 4 个 i，把 residual 的 4 份留在寄存器/LDS
里**，这样 residual 只读一遍。这是本题的主要判据，但题面不会说出来。

反过来说，如果实现按 i 分工、靠 L2 兜住重复读，M=1024 时 residual 只有
33.6 MB，L2 是 8 MB，兜不住 —— 会看到带宽随 M 变化，那不是 DPM 抖动。

## 数值

`residual` 用 fp32 累加。bf16 直接累加 4 份，和 4 个值相互抵消时相对误差
会爆表（同 `moe-sum-l1` 的理由），所以走 fp32 中间量。
"""
import torch


def arun(x: torch.Tensor, residual: torch.Tensor,
         post: torch.Tensor, comb: torch.Tensor):
    """返回 out[M, HC, H] bf16。

    x:        [M, H]       bf16
    residual: [M, HC, H]   bf16
    post:     [M, HC]      fp32
    comb:     [M, HC*HC]   fp32，行主序：comb[m, i*HC + j]
    """
    HC = residual.shape[1]
    xf = x.float()                                   # [M, H]
    rf = residual.float()                            # [M, HC, H]
    pf = post.float()                                # [M, HC]
    cf = comb.float().view(-1, HC, HC)               # [M, HC, HC]

    # ① post[m, i] * x[m, h] → [M, HC, H]
    out = pf[:, :, None] * xf[:, None, :]
    # ② Σ_j comb[m, i, j] * residual[m, j, h]
    out = out + torch.einsum("mij,mjh->mih", cf, rf)
    return out.to(torch.bfloat16)
