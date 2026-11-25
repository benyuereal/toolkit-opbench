"""参考实现（正确性口径）。纯 torch，不依赖被测库。

## 语义

MoE 计算前，token 必须按专家重排成**块对齐**的布局，下游 grouped GEMM
才能按固定 tile 高度切分。

    topk_ids[M, K]      每个 token 选中的 K 个专家 id，取值范围 [0, E)
    ALIGN               每个专家占用的 slot 数必须是 ALIGN 的倍数

    sorted_ids[CAP]     重排后的 token 下标（token*K + 它在 topk 里的位置）
    expert_ids[BLOCKS]  第 b 个块属于哪个专家
    num_post_pad        前多少个 sorted_ids 是有效的
    CAP     = M*K + E*ALIGN
    BLOCKS  = CAP // ALIGN

## 三个容易错的地方

1. **每个专家单独 pad 到 ALIGN 的倍数**，不是整体 pad。专家 0 的尾部空格
   不能借给专家 1 用。
2. **只有前 `num_post_pad` 个 sorted_ids 有效**。后面的内容是未初始化/脏的，
   逐位比对时会不一致——这是库实现的行为，不是它算错了。
   所以正确性只比 `[:num_post_pad]` 与 `expert_ids[:ceil(npd/ALIGN)]`。
3. 排序**不要求稳定**：同一专家内部 token 的先后顺序由实现决定。
   拿"逐位相等"判 sorted_ids 会把正确实现判错；要比**集合**。
"""
import torch


def arun(topk_ids, align: int = 128, num_experts: int = 64):
    """返回 (sorted_ids, expert_ids, num_post_pad)。

    参数顺序与 `lightop.moe_align_block_size(topk_ids, block_size, num_experts)`
    一致（block_size 在前），这是被基线实现约束的。
    """
    dev = topk_ids.device
    M, K = topk_ids.shape
    E = num_experts

    flat = topk_ids.reshape(-1)                      # [M*K]
    # 每个专家的 token 数
    counts = torch.bincount(flat, minlength=E)       # [E]
    # 每个专家 pad 到 align 的倍数
    padded = ((counts + align - 1) // align) * align
    # 每个专家在 sorted 里的起点（累加前面所有专家的 padded 数）
    offsets = torch.cumsum(padded, 0) - padded       # [E]

    npost = int((offsets[-1] + padded[-1]).item())

    # 稳定的分组排序：按 (expert, 原位置) 排，得到每个专家内部的连续段
    order = torch.argsort(flat, stable=True)         # 按专家号排
    sorted_ids = order.to(torch.int32)

    # expert_ids：第 b 块（ALIGN 个 slot）属于哪个专家
    nblocks = npost // align if npost % align == 0 else npost // align + 1
    blk2exp = torch.zeros(nblocks, dtype=torch.int32, device=dev)
    for e in range(E):
        start = int(offsets[e].item())
        end = start + int(padded[e].item())
        blk2exp[start // align:(end + align - 1) // align] = e

    return sorted_ids, blk2exp, torch.tensor([npost], dtype=torch.int32, device=dev)
