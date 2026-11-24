"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

语义（与 flash_mla.flash_mla_sparse_fwd 对齐）：
  score[i,h,j] = q[i,h,:] · kv[indices[i,0,j],:] * sm_scale    j ∈ [0, topk)
  out[i,h,:]   = softmax_j(score) · kv[indices[i,0,j], :d_v]

**三个必须对齐的点**：
1. `indices == -1`（或 >= SKV）表示该槽位无效，softmax 前要掩成 -inf，
   **不能当成索引 0**。生产里 padding 比例不低。
2. `sm_scale = 1/sqrt(d_qk)`（d_qk=576 时是 1/sqrt(576)），
   不是 1/sqrt(640) —— 640 是 latent+rope 之外的另一个口径。
3. 输出只有 d_v=512 维（latent 部分），后 64 维 rope tail **只参与打分不输出**。
"""
import math
import torch


def arun(q, kv, indices, sm_scale: float | None = None, d_v: int = 512):
    S, H, D = q.shape
    topk = indices.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    kvf = kv[:, 0, :].float()           # [SKV, D]
    qf = q.float()                       # [S, H, D]
    idx = indices[:, 0, :].long()        # [S, topk]
    SKV = kvf.shape[0]

    valid = (idx >= 0) & (idx < SKV)     # -1 与越界都无效
    safe = idx.clone()
    safe[~valid] = 0
    gathered = kvf[safe]                 # [S, topk, D]

    score = torch.einsum("shd,std->sht", qf, gathered) * sm_scale
    score = score.masked_fill(~valid[:, None, :], float("-inf"))
    # 全无效的行：softmax 会得到 nan；按 0 处理（与 kernel 的空行语义一致）
    all_invalid = (~valid).all(dim=-1)
    score = torch.nan_to_num(score, nan=0.0, neginf=-1e30)
    p = torch.softmax(score, dim=-1)
    out = torch.einsum("sht,std->shd", p, gathered[:, :, :d_v])
    return out.to(q.dtype)
