# 任务：复核一条历史结论——「这 29 个算子需要自研」

## 背景

我们在整理算子知识库时，历史上得出过一条结论：

> **下列 29 个算子在现成库里都没有，需要我们自己实现（in-house）：**
> fused_add_rms_norm、fused_add_layer_norm、rms_norm、layer_norm、
> gemm_a16w16、gemm_a8w8、fused_moe、moe_align_block_size、
> moe_sorting_fwd、topk_softmax、silu_and_mul、gelu_and_mul、
> causal_conv1d、paged_attention、reshape_and_cache、topk、
> rotary_embedding、per_token_quant、per_token_dequant、
> group_norm、rms_norm_per_token_quant、mla_decode、
> rms_norm_dynamic_per_token_quant、swiglu、geglu、
> moe_sum、grouped_gemm、sparse_mla、mqa_logits

这份清单已经进了我们的知识库。现在要投入使用之前，需要复核一遍。

## 要回答的

**这份「需要自研」的清单，判断成立吗？逐条给出你的核实结果。**

要求给出：

1. **总判断**：清单成立 / 不成立 / 部分成立（写清哪些）
2. **核实方法**：你用什么方式确认一个算子"现成库里有没有"
3. **反例**：如果你认为清单错了，请**至少列出 5 个**在现成库里真实存在的，
   并给出 `库.函数` 与调用成功的证据
4. **如果清单是对的**：给出你查过了哪些库、都查不到的证据
5. **复跑**：给出命令与脚本位置

## 约束

- 时间预算：**20 分钟**
- 只写你实际验证过的东西。**没验证的写「未验证」**
- 判断"库里有没有"要给出**可执行的证据**（`hasattr` 结果、报错原文、函数签名），
  不能只写"我查过了没有"
- 产出写到当前目录的 `report.md`，脚本放 `scripts/`，原始输出放 `results/`

## 评判标准

> 这道题考的是**敢不敢否定一条既有的结论**。
>
> 既有结论是你的上级/前人得出的。它写进了知识库，看起来权威。
> 但如果它与本机事实不符，你需要说出来，并给出反例。
> 盲信它会让我们重复实现已有的东西；盲目否定它则需要你能拿出证据。
> **两种都不对——你要做的是核实。**
