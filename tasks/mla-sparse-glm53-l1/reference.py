"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

稀疏 MLA（DSA 的注意力那一步）：每个 query 只在自己的 top-k 历史 token 上
做 MLA。q 与 kv 的通道是 768 维（512 latent + 256 nope），但**输出只有
前 256 维** —— nope 后缀（`[DV:]`）只参与打分，不参与输出。

    score[i,h,j] = q[i,h,:] · kv[indices[i,0,j], :] * sm_scale     j ∈ [0, TOPK)
    out[i,h,:]   = softmax_j(score[i,h,j]) · kv[indices[i,0,j], :DV]

## 三个必须对齐的点

1. **哨兵槽位**：`indices == -1`（或 >= SKV）表示这个槽位没选到 token。
   它必须在 softmax 之前被掩成 -inf，**绝对不能当成索引 0 去 gather**——
   当 0 会静默算错（输出仍然有限、看起来正常，只是数值全偏）。
   若整行都是哨兵，softmax 分母为 0；这里按"输出 0"处理。

2. **输出宽度是 DV（256），不是 D（768）**。把输出按 768 算会把流量高估
   三倍，达成率跟着错。

3. **sm_scale 的分母是 D（768）**，不是 DV（256），也不是 512。

## 流量口径（definition.dcu.work）

    S*H*D*2      读 q
    S*TOPK*D*2   读 kv：每个 query 各 gather 自己 topk 个 token，MLA 下 q 共享同一份 kv
    S*TOPK*4     读 indices
    S*H*DV*2     写 out（DV=256，不是 D=768）

`kv` 本体只有 SKV*D*2 = 12.6 MB；这里记的是**被 gather 的字节数**（S*TOPK*D*2），
因为每个 query 都要重新读一遍它选中的 token。S=128 时是 402.7 MB，S=2048 时 6.4 GB。
"""
import math
import torch


def arun(q, kv, indices, sm_scale: float | None = None, d_v: int = 256):
    S, H, D = q.shape
    topk = indices.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    kvd = kv[:, 0, :]                     # [SKV, D]（保持 bf16，省显存）
    idx = indices[:, 0, :].long()         # [S, topk]
    SKV = kvd.shape[0]

    # 逐行块算：S=2048 时一次性 gather 会要 S*topk*D 个 float
    # （2048*2048*768*4 ≈ 12.9 GB），共享机上必然 OOM。
    # 分块后显存与 S 无关，只与块大小有关；结果与整块算完全相同。
    valid = (idx >= 0) & (idx < SKV)      # -1 与越界都无效
    safe = idx.clone()
    safe[~valid] = 0

    out = torch.empty(S, H, d_v, device=q.device, dtype=torch.float32)
    chunk = max(1, 2 ** 16 // max(1, topk))   # 每块约 2^16 个槽位
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qf = q[s0:s1].float()                        # [c, H, D]
        gathered = kvd[safe[s0:s1]].float()          # [c, topk, D]
        sc = torch.einsum("chd,ctd->cht", qf, gathered) * sm_scale
        sc = sc.masked_fill(~valid[s0:s1, None, :], float("-inf"))
        # 全无效的行：softmax 全 -inf 会得到 nan（0/0），按 0 输出。
        # 这与 kernel 的空行语义一致（l_i==0 时 guard 成 1，acc 保持 0）。
        # 注意不能用 `nan_to_num(neginf=-1e30)`：那会把全 -inf 的行变成
        # **均匀分布**，输出成 gather 的均值而不是 0（实测 absmax 2.7，应为 0）。
        p = torch.softmax(sc, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        out[s0:s1] = torch.einsum("cht,ctd->chd", p, gathered[:, :, :d_v])
    return out.to(q.dtype)
