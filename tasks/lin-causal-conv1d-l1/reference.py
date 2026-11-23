"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

语义必须与 causal_conv1d.causal_conv1d_fn(x, weight, bias=None) 一致：

    out[b, d, t] = sum_{k=0}^{W-1} weight[d, k] * x[b, d, t - (W - 1) + k]

即「只在 t 及之前取值」的 causal 短卷积，序列开头按 0 填充；
输出与 x 同形状 (B, D, S)。

实现用 depthwise conv1d（padding=W-1 后截断），与上游 reference
（causal_conv1d_ref）同一写法。中间量用 float32：bf16 直接累加 W=4 项
会有可见误差，那种误差会被误判成"被测实现算错了"。

注意：这里只定义**正确性**，不是性能基线。
"""
import torch
import torch.nn.functional as F


def arun(x, weight):
    dtype = x.dtype
    B, D, S = x.shape
    W = weight.shape[1]
    y = F.conv1d(
        x.float(), weight.float().unsqueeze(1), None,
        padding=W - 1, groups=D,
    )
    return y[..., :S].to(dtype)
