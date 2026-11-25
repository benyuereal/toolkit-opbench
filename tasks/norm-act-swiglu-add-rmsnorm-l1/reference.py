"""参考实现（正确性口径）。纯 torch，不依赖被测库。

融合语义（三步合一）：
    xf  = x + residual                      # add
    h   = w * xf / sqrt(mean(xf²) + eps)    # rmsnorm
    g, u = h.chunk(2, dim=-1)               # swiglu 的 gate/up 拆分
    out = silu(g) * u                       # silu-and-mul

**关键语义点**：
1. `eps` 加在 sqrt **内**（sqrt(mean+eps)），与 lightop.RMSNormAdd 一致。
2. gate/up 沿**最后一维对半切**：h 的宽度是 2*I，前 I 是 gate，后 I 是 up。
3. 输出宽度是 I（不是 2I）——两个半张量乘成一个。

**流量口径**（definition.dcu.work = `2*M*H*2 + M*I*2`）：
    读 x, residual:  2 * M*H*2   = 134.2 MB
    写 out:          1 * M*I*2   =  33.6 MB
    合计                             167.8 MB   （M=8192, H=4096, I=2048）
    weight 走 L2，忽略。
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
