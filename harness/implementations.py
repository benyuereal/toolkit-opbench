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


def lightop_moe_align(topk_ids, block_size: int = 128, num_experts: int = 64):
    """lightop 的 moe_align_block_size（本机唯一现成路径）。

    签名与直觉相反：`(topk_ids, block_size, num_experts)`——**block_size 在前**。
    实测按 `(topk_ids, num_experts, block_size)` 传会抛
    `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`，
    那个报错和参数顺序毫无关系，很容易被误导去查 dtype。

    返回 (sorted_ids, expert_ids, num_tokens_post_pad)，缓冲区由库内部分配。
    """
    import lightop

    return lightop.moe_align_block_size(topk_ids, block_size, num_experts)


def lightop_per_token_quant_fp8(x):
    """lightop 的 per-token FP8 动态量化（纯 kernel，无 weight）。

    `lightop.per_token_quant_fp8` 只在子模块 `lightop.quant` 下，**不在顶层**
    （`hasattr(lightop, "per_token_quant_fp8")` 是 False，2026-11-25 实测）。
    顶层有 `rms_norm_per_token_fp8_quant`，但那个要 weight 参数、做的是
    rmsnorm+quant 两步，不是纯量化。

    签名 `(x, dtype=float8_e4m3fn, out_q=None, out_scale=None)`。
    不传 out_* 时库内部 `torch.empty_like` 分配——这就是生产写法（调用方
    通常也没法预分配，因为 M 来自 batch）。计时口径按这个形态。

    返回 (q[M,H] fp8_e4m3, scale[M,1] fp32)。
    """
    from lightop import quant as lq

    return lq.per_token_quant_fp8(x)


def aiter_per_token_quant_hip(x):
    """aiter 的 per-token 量化 HIP kernel，作为对照实现。

    `aiter.per_token_quant_hip(x, scale=None, quant_dtype=..., num_rows=None, ...)`。
    scale 传 None 时内部 `torch.empty((*shape[:-1], 1), fp32)`。
    它会调 `dynamic_per_token_scaled_quant`，是同一套 C++ 入口。

    **已知环境缺陷（不要当基线）**：本机偶发
    `NotImplementedError: "min_all_cuda" not implemented for 'Float8_e4m3fn'`
    （aiter 0.1.6）。实测 200 次连调 0 失败，但换 seed / 换张量后会零星触发，
    触发一次就中断整轮计时。所以基线用 lightop。
    """
    import torch

    import aiter

    return aiter.per_token_quant_hip(x, quant_dtype=torch.float8_e4m3fn)


def aiter_moe_sum(expert_out):
    """aiter 的 moe_sum：把 K 份专家输出求和回 token。

    schema: `aiter.moe_sum(input[M, K, H], output[M, H]) -> ()`
    **原地写 output**，两参数、无返回值。

    实测踩过：按 `(input, output, topk_indices)` 三参数调会抛
    `expected at most 2 argument(s) but received 3`——签名是 2 个，
    不像 topk_softmax 那样要三个缓冲区。
    """
    import aiter
    import torch

    M, K, H = expert_out.shape
    out = torch.empty(M, H, dtype=expert_out.dtype, device=expert_out.device)
    aiter.moe_sum(expert_out, out)
    return out


def sglang_mhc_post(x, residual, post, comb):
    """GLM-5.3-Flash 的 mHC 后处理，走 sglang 自带的 tilelang 实现。

        out[m, i, h] = post[m, i] * x[m, h]
                     + Σ_j comb[m, i*4 + j] * residual[m, j, h]

    ## 签名对不上的地方

    `mhc_post_tilelang(a, b, c, d, x, hc, hidden, n_thr=128, h_blk=1024)`——
    位置参数有 9 个，其中**第 5 个才是要写的 output 缓冲区**，前四个是
    `comb.view(-1,4,4)` / `residual` / `post.squeeze(-1)` / `x`。这个顺序
    在 sglang 源码里是 `_post[m,4,cdiv(h,256)](x,residual,post,comb,o,h,256)`，
    和 Python 包装器的参数顺序**不一致**，不要照着 kernel 的调用抄。

    `b_post`（同名）是另一个不复用缓冲区的版本，题面不给它开洞：本题的
    基线是**带缓冲区的生产路径**，这是 mHC 在服务里真实的调用形态。

    ## 计时口径

    `torch.empty_like(residual)` 在计时窗口之外——那是调用者的分配，不是
    这个 kernel 的成本。这与 `lightop_add_rmsnorm_swiglu` 的口径一致。

    ## 为什么拿它当基线

    本机实测默认参数下它跑到 749 GB/s（m=4096，9H 口径），而同一份实现
    只调 `n_thr`/`h_blk` 就能到 1152 GB/s（1.54x），同卡纯拷贝是 1493 ——
    说明它的分块对形状很敏感。这不是"实现写错了"，是**参数没跟着形状走**，
    正好是 L1 题该暴露的。
    """
    import torch

    from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang

    M, HC, H = residual.shape
    out = torch.empty_like(residual)
    mhc_post_tilelang(comb.view(-1, HC, HC), residual, post.squeeze(-1),
                      x, out, HC, H)
    return out


def sglang_sparse_mla_glm53(q, kv, indices, sm_scale: float = 1.0 / 768**0.5,
                            d_v: int = 256):
    """GLM-5.3-Flash 形状的稀疏 MLA 基线（sglang 自带的 DSA Triton kernel）。

    ## 为什么落在这里

    `flash_mla`（1.2.0）**吃不下本题形状**，两条路都堵死（2026-11-25 实测）：

      * `flash_mla_sparse_fwd(q[H,64,768], ...)` → `RuntimeError: Invalid d_qk: 768`
        （prefill kernel 只接受 d_qk ∈ {512, 576}，且 `d_v` 只能是 512）
      * `flash_mla_with_kvcache(..., indices=...)` → `RuntimeError: DSA BF16 sparse
        decode only supports d_qk == 512 for now`

    `aiter.ops.tilelang.sparse_mla_fwd` 能编译（dim=256/tail_dim=512），但它内部
    带 causal 掩码：随机 top-k 下标（非因果）下 max_abs 0.2–0.57，是错的结果，
    不能当基线。

    可用且语义正确（随机下标下 max_abs ≈ 2e-3）的是 sglang 的
    `sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd`。
    这也是本机 GLM 服务真实走的 DSA 路径上游。

    ## 签名与形状约定（原样转发，不做任何预处理）

        triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=256)
          q_nope:  [S, H, d_v]       q 的前 d_v 维
          q_rope:  [S, H, D-d_v]     q 的后 512 维（本题 D=768 → 768-256）
          kv:      [SKV, 1, D]       bf16
          indices: [S, 1, TOPK]     int32，-1 表示无效槽位
          → 返回 [1, S, H, d_v] bf16

    注意它把 q **拆成两段**吃：主段宽 d_v、尾段宽 D-d_v。这与题面「q 是
    一整块 [S,64,768]」不同，适配层在这里做切片是**参数形状对齐**，
    不是优化。

    ## 计时口径

    `q_nope.contiguous()` / `q_rope.contiguous()` 在计时窗口内执行。理由：
    切片出来的 `q[..., :256]` 本身不连续，生产里也不会有预先把 q 拆成两块
    的免费机会。把它放到窗口外，量到的就不是"跑这个算子"的代价。

    num_heads=64 恰好是该 kernel `H` 维的一个 warp-tile，不需要 padding。
    """
    import torch

    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
        triton_sparse_mla_fwd,
    )

    q_nope = q[..., :d_v].contiguous()
    q_rope = q[..., d_v:].contiguous()
    out = triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v)
    return out[0]


def sglang_sparse_mla_glm53_fp8(q, kvq, kscale, indices,
                                sm_scale: float = 1.0 / 768**0.5,
                                d_v: int = 256):
    """GLM-5.3-Flash 形状 + FP8 KV 的稀疏 MLA 基线。

    ## 这是降级路径，不是最优路径

    本机**没有**能吃 GLM 形状（D=768 / DV=256）的 FP8 稀疏 MLA kernel
    （2026-11-26 逐个试过）：

      * `flash_mla.flash_mla_sparse_fwd`   → `Invalid d_qk: 768`
      * `flash_mla.flash_mla_with_kvcache_fp8` → 只支持 d_qk=512/576、d_v=512
      * `aiter.ops.tilelang.sparse_mla_fwd` → 传 fp8 输入报
        `ufunc_add_CUDA not implemented for Float8_e4m3fn`

    所以基线只能是生产里真实的降级做法：**先把 FP8 KV 反量化成 bf16，
    再走已验证正确的 bf16 路径**（sglang 的 DSA Triton kernel）。

    这个降级本身就是本题要测的东西——它多了一趟全量反量化，
    而反量化只要覆盖被 gather 的那部分就够。**优化空间就在这里**，
    但题面不会这么说。

    ## 反量化的位置

    整块反量化 `kvq[8192,768]` 只有 6.3 MB，比逐 gather 反量化
    （S=128 时 201 MB 的 fp8 → 402 MB 的 bf16）省得多。所以基线选择
    **整块反量化一次**，这是降级路径里最合理的写法，也保证我们给 agent
    的基线不是故意写坏的。

    ## 计时口径

    反量化在计时窗口内——它就是这条路径的代价。放在窗口外量到的
    只是 bf16 kernel 的耗时，那和"用 FP8 KV 省带宽"这个目标无关。
    """
    import torch

    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
        triton_sparse_mla_fwd,
    )

    TILE = 128
    D = kvq.shape[-1]
    n = D // TILE
    # 整块反量化：[SKV, 1, D] fp8 × [SKV, 1, n] fp32 → bf16
    s = kscale[:, :, :n].to(torch.float32).repeat_interleave(TILE, dim=-1)
    kvd = (kvq.to(torch.float32) * s).to(torch.bfloat16).contiguous()

    q_nope = q[..., :d_v].contiguous()
    q_rope = q[..., d_v:].contiguous()
    out = triton_sparse_mla_fwd(q_nope, q_rope, kvd, indices, sm_scale, d_v)
    return out[0]
