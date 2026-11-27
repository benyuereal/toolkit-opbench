"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

稀疏 MLA（DSA 的注意力那一步），**KV 以 FP8 存储**。每个 query 只在
自己的 top-k 历史 token 上做 MLA。

    kv_fp32[t, c] = kvq[t, c].to(fp32) * kscale[t, c // 128]

    score[i,h,j] = q[i,h,:] · kv_fp32[indices[i,0,j], :] * sm_scale
    out[i,h,:]   = softmax_j(score[i,h,j]) · kv_fp32[indices[i,0,j], :DV]

q 与 kv 的通道是 768 维（512 latent + 256 nope），但**输出只有前 256 维** ——
nope 后缀只参与打分，不参与输出。

## 四个必须对齐的点

1. **反量化的 scale 粒度是 128 个通道**，不是整个 token、也不是逐通道。
   第 `c` 个通道用 `kscale[t, c // 128]`，768 维共 6 个 scale。
   用错粒度的后果：逐 token（1 个 scale）= 高估精度、数值偏离；
   逐通道（768 个 scale）= 形状对不上、直接报错。

2. **哨兵槽位**：`indices == -1`（或 >= SKV）表示没选到 token。
   必须在 softmax 之前掩成 -inf，**不能当成索引 0 去 gather**——
   当 0 会静默算错（输出仍然有限、看起来正常，只是数值全偏）。
   整行都是哨兵时按"输出 0"处理。

3. **输出宽度是 DV（256），不是 D（768）**。按 768 算会把流量高估。

4. **sm_scale 的分母是 D（768）**，不是 DV，也不是 512。

## 流量口径（definition.dcu.work）

    S*H*D*2           读 q（bf16）
    S*TOPK*D          读 kvq（**fp8 = 1 字节**，这是 FP8 相对 bf16 省一半的地方）
    S*TOPK*QTILE*4    读 kscale（每个被 gather 的 token 带 6 个 fp32）
    S*TOPK*4          读 indices
    S*H*DV*2          写 out

**FP8 的收益就在这里**：kv 项从 `S*TOPK*D*2` 变成 `S*TOPK*D`，正好减半。
但 scale 项要额外读 `S*TOPK*24` 字节（6 个 fp32），约占 kv 项的 3.1%。

S=128：q 12.58 + kv 201.33 + scale 6.29 + ind 1.05 + out 4.19 = 225.4 MB
S=2048：q 201.33 + kv 3221.23 + scale 100.66 + ind 16.78 + out 67.11 = 3607.1 MB

对比 bf16 版（`mla-sparse-glm53-l1`）：S=128 时 420.5 MB → 225.4 MB（省 46%）。
"""
import math
import torch

TILE = 128          # 量化粒度：每 128 个通道一个 scale


def dequant(kvq: torch.Tensor, kscale: torch.Tensor) -> torch.Tensor:
    """[..., D] fp8 + [..., D/128] fp32 → [..., D] float32。

    单独抽出来是为了让"粒度是 128"这件事在代码里只出现一次。
    """
    D = kvq.shape[-1]
    n = D // TILE
    # [..., n] -> [..., n, TILE] -> 展平回 [..., D]
    s = kscale[..., :n].to(torch.float32).repeat_interleave(TILE, dim=-1)
    return kvq.to(torch.float32) * s


def arun(q, kvq, kscale, indices, sm_scale: float | None = None, d_v: int = 256):
    S, H, D = q.shape
    topk = indices.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    idx = indices[:, 0, :].long()          # [S, topk]
    SKV = kvq.shape[0]

    valid = (idx >= 0) & (idx < SKV)       # -1 与越界都无效
    safe = idx.clone()
    safe[~valid] = 0

    out = torch.empty(S, H, d_v, device=q.device, dtype=torch.float32)
    # 逐块算：S=2048 时一次性 gather 要 S*topk*D 个 float32
    # （2048*2048*768*4 ≈ 12.9 GB），共享机上必然 OOM。
    # 分块后显存与 S 无关；结果与整块算完全相同。
    chunk = max(1, 2 ** 16 // max(1, topk))
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qf = q[s0:s1].float()                          # [c, H, D]
        si = safe[s0:s1]                               # [c, topk]（已是 2 维）
        gq = kvq[si, 0, :]                             # [c, topk, D] fp8
        gs = kscale[si, 0, :]                          # [c, topk, n]
        gathered = dequant(gq, gs)                     # [c, topk, D] fp32

        sc = torch.einsum("chd,ctd->cht", qf, gathered) * sm_scale
        sc = sc.masked_fill(~valid[s0:s1, None, :], float("-inf"))
        # 全无效的行：softmax 全 -inf 会得到 nan（0/0），按 0 输出。
        # 注意不能用 `nan_to_num(neginf=-1e30)`：那会把全 -inf 的行变成
        # **均匀分布**，输出成 gather 的均值而不是 0。
        p = torch.softmax(sc, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        out[s0:s1] = torch.einsum("cht,ctd->chd", p, gathered[:, :, :d_v])
    return out.to(q.dtype)
