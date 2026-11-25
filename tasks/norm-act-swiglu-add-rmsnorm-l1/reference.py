"""参考实现（正确性口径）。纯 torch，不依赖被测库。

融合语义（三步合一）：
    xf  = x + residual                      # add
    h   = w * xf / sqrt(mean(xf²) + eps)    # rmsnorm
    g, u = h.chunk(2, dim=-1)               # swiglu 的 gate/up 拆分
    out = silu(g) * u                       # silu-and-mul

**关键语义点（评分会按这三条判）**：
1. `eps` 加在 sqrt **内**（sqrt(mean+eps)），与 lightop.RMSNormAdd 一致。
2. gate/up 沿**最后一维对半切**：h 的宽度是 2*I，前 I 是 gate，后 I 是 up。
3. 输出宽度是 I（不是 2I）——两个半张量乘成一个。

**流量口径（2 读 2 写之外的额外读写）**：
    读 x, residual: 2 * M*H*2
    写 y(融合后的中间，若实现需要): M*2I*2
    写 out:          M*I*2
    weight 走 L2，忽略
"""
import math
import torch
import torch.nn.functional as F


def arun(x, residual, weight, eps: float = 1e-6):
    xf = x.float() + residual.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    h = (h * weight.float()).to(x.dtype)
    g, u = h.chunk(2, dim=-1)
    out = F.silu(g.float()) * u.float()
    return out.to(x.dtype)
