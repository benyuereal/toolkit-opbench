"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

MoE 每个 token 被 K 个专家各算一遍，最后要把 K 份专家输出**加权求和**
回该 token（这里权重已在外层乘进输入，本题只做求和）。

    expert_out[M, K, H]  →  out[M, H] = sum_k expert_out[:, k, :]

## 一个容易错的地方

沿 **K 轴**（第 1 维）求和，不是沿 H。沿 H 求和会得到一个 [M, K] 张量，
形状对不上但某些框架会静默广播。

累加用 fp32：bf16 直接累加 K=8 份会在量级差异大时丢精度，
那不是"实现算错了"，是精度口径不同。
"""
import torch


def arun(expert_out):
    return expert_out.float().sum(dim=1).to(expert_out.dtype)
