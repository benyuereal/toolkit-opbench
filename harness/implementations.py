"""厂商实现的薄适配器，供 `definition.dcu.baseline_impl` 以点分路径引用。

## 为什么需要这一层

`baseline.build()` 的调用约定是 `impl(*按 inputs 键序的张量, *extra_args)`。
但厂商实现的签名五花八门：

  * `aiter.moe_c_topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)`
    前三个是**输出缓冲区**，必须预先分配；调用后结果写在缓冲区里。
  * `lightop.fused_add_rms_norm(x, residual, w, eps)` 原地改写，返回元组。

把这些适配写进题目文件会让题面变成"某一道题的程序"；写进 baseline.py
又会让那层泛型代码认识具体算子的参数表。所以单独一层：**这里可以有
算子知识，baseline 层保持泛型**。

## 纪律

* 每个适配器只做"参数形状对齐"，不做任何优化或预处理。
* 输出缓冲区每次调用重新分配——把它挪到外面会让计时漏掉分配开销，
  但我们本来就不该把那部分算进算子耗时。这一点在各自的 docstring 里说明。
"""
from __future__ import annotations


def aiter_topk_softmax(gating_output, top_k: int = 8):
    """aiter 的 topk_softmax 适配器。

    schema: (topk_weights, topk_indices, token_expert_indices, gating_output)
    前三个是输出/工作缓冲区，必须预分配；`token_expert_indices` 的形状是
    M*E 的 int32（实测它填的是「第 i 个 token 的第 j 个专家」的扁平索引）。

    缓冲区在内部分配，不计入算子耗时——那是调用者的事，不是 kernel 的事。
    """
    import torch

    import aiter

    M, E = gating_output.shape
    weights = torch.empty(M, top_k, dtype=torch.float32, device=gating_output.device)
    indices = torch.empty(M, top_k, dtype=torch.int32, device=gating_output.device)
    token_expert = torch.empty(M * E, dtype=torch.int32, device=gating_output.device)
    aiter.moe_c_topk_softmax(weights, indices, token_expert, gating_output)
    return weights, indices


def aiter_topk_softmax_buf(gating_output, top_k: int = 8):
    """同 aiter_topk_softmax，但缓冲区按形状缓存复用。

    用于对比「分配开销占多少」——两个版本都存在于本文件，是为了让
    这类差异**可测量**，而不是靠猜。
    """
    import torch

    import aiter

    M, E = gating_output.shape
    key = (M, E, top_k, gating_output.device.index)
    cache = aiter_topk_softmax_buf._cache
    if key not in cache:
        cache[key] = (
            torch.empty(M, top_k, dtype=torch.float32, device=gating_output.device),
            torch.empty(M, top_k, dtype=torch.int32, device=gating_output.device),
            torch.empty(M * E, dtype=torch.int32, device=gating_output.device),
        )
    weights, indices, token_expert = cache[key]
    aiter.moe_c_topk_softmax(weights, indices, token_expert, gating_output)
    return weights, indices


aiter_topk_softmax_buf._cache = {}


def lightop_add_rmsnorm_swiglu(x, residual, weight, eps: float = 1e-6):
    """fused_add_rms_norm + fuse_silu_and_mul 两步组合（基线实现）。

    本机 lightop 0.6.0 / aiter 0.1.6 没有三合一入口（2026-11-25 已核）。

    **原地改写**：`fused_add_rms_norm` 把结果写回 x 和 residual。
    调用方每轮必须喂独立副本（definition 里已声明 inplace_inputs）。

    签名顺序与 definition.inputs 一致：(x, residual, weight) + eps。
    """
    import torch

    import lightop

    lightop.fused_add_rms_norm(x, residual, weight, eps)
    out = torch.empty(x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device)
    lightop.fuse_silu_and_mul(x, out)
    return out
