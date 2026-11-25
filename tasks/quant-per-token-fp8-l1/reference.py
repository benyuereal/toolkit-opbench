"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

LLM 推理里，线性层的激活要量化成 FP8 后再送进 W8A8 GEMM。**per-token**（每行
一个 scale）是主流做法：一行 token 的动态范围在行内相近，但不同 token 之间
可以差好几个数量级，共用一个 scale 会让小 token 整个塌成 0。

    amax[i]  = max_j |x[i, j]|                      # 每行的最大绝对值
    scale[i] = amax[i] / 448.0                      # fp8_e4m3 的最大可表示值
    q[i, j]  = cast_to_fp8_e4m3(x[i, j] / scale[i])  # round-to-nearest-even，

**三个必须对齐的点**：

1. **per-token，不是 per-tensor**：`scale` 沿**行**（最后一维）归约，形状 `[M, 1]`。
   per-tensor 是把整个张量归约成一个标量——那是另一种量化。
2. **quant_max = 448.0**：`float8_e4m3fn` 的最大规格化值。写成 127 / 128 / 255
   都对应别的量化格式，数值对不上。
3. **scale 是 fp32**：`[M, 1]`、`torch.float32`。bf16 存 scale 会额外丢 8 位尾数，
   反量化误差会明显变大。

**全零行**：`amax = 0` 时不能除零。约定 `scale = 1.0`，`q` 全 0。
本机 `aiter` 与 `lightop` 都按这个约定实现（2026-11-25 实测）。

**round 方向**：fp8 用硬件 cast（round-to-nearest-even），不做截断。截断会让
每一行的相对误差系统性偏大约 0.1–0.2 个百分点（实测 rel_l2 0.0265 → 0.0271）。

## 流量口径（definition.dcu.work = `M*H*2 + M*H + M*4`）

    读 x:      M*H*2   bf16
    写 q:      M*H*1   fp8
    写 scale:  M*4     fp32
    M=8192, H=4096 时合计 100.7 MB
"""
import torch

QUANT_MAX = 448.0  # float8_e4m3fn 的最大规格化值


def arun(x: torch.Tensor):
    """输入 x[M, H] bf16，返回 (q[M, H] fp8_e4m3, scale[M, 1] fp32)。"""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)          # [M, 1]
    scale = amax / QUANT_MAX                            # [M, 1]
    # 全零行：scale=0 会除零，约定用 1.0（q 反正是 0）
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (xf / scale).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)
