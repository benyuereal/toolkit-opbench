"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

语义（与 aiter.moe_c_topk_softmax 对齐）：
  probs = softmax(gating_output, dim=-1)
  topk_weights, topk_indices = topk(probs, k)

注意 topk_weights 是**未归一化**的 softmax 概率（不除以 top-k 之和），
这一点必须与 kernel 一致，否则数值对不上。
"""
import torch


def arun(gating_output, top_k: int = 2):
    probs = torch.softmax(gating_output.float(), dim=-1)
    w, idx = torch.topk(probs, top_k, dim=-1)
    return w, idx.to(torch.int32)
