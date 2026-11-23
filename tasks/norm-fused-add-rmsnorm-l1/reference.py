"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

语义必须与 lightop.fused_add_rms_norm 一致：
  h   = x + residual                      （先相加，bf16 精度下按 fp32 累加）
  y   = h / sqrt(mean(h^2) + eps) * w
返回 ((y, h)) —— 第二个输出是**相加后的残差**，不是输入 residual 本身。

这里用 float32 中间量：bf16 直接算 mean(h^2) 会有明显误差，
那种误差会被误判成"被测实现算错了"。
"""
import torch


def arun(x, residual, weight, variance_epsilon: float = 1e-6):
    h = (x.float() + residual.float())
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + variance_epsilon)
    y = (y * weight.float()).to(x.dtype)
    return y, h.to(x.dtype)
