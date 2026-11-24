"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

语义（与 aiter.moe_c_topk_softmax 对齐）：
  probs = softmax(gating_output, dim=-1)
  topk_weights, topk_indices = topk(probs, k)

注意 topk_weights 是**未归一化**的 softmax 概率（不除以 top-k 之和），
这一点必须与 kernel 一致，否则数值对不上。

**top_k 的默认值必须与 definition 的 TOPK 轴一致（8）**，不能写 2。
`bench.check.correctness.check_all` 按 `arun(*按 inputs 键序的张量)` 调用，
不会从 definition 里读 TOPK 传进来——所以默认值必须是对的，否则
"标准答案"本身就错，三臂全被判 fail。
"""
import torch

from bench.data import definition as _defn


def _default_topk() -> int:
    """从题目的 definition 读 TOPK。[moe-gating-l1] 里它是 const 轴。"""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    d = _defn.load(os.path.join(here, "definition.json"))
    return int(d.const_axes().get("TOPK", 8))


def arun(gating_output, top_k: int | None = None):
    if top_k is None:
        top_k = _default_topk()
    probs = torch.softmax(gating_output.float(), dim=-1)
    w, idx = torch.topk(probs, top_k, dim=-1)
    return w, idx.to(torch.int32)
