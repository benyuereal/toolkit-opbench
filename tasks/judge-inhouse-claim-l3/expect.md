# 参考答案（不给 agent 看）

## 结论：**清单不成立。绝大多数算子现成库里都有。**

本机实测（`lightop` / `aiter` / `causal_conv1d` / `flash_attn` 等已安装库）：

| 清单项 | 实际所在的库 |
|---|---|
| fused_add_rms_norm | `lightop.fused_add_rms_norm` |
| fused_add_layer_norm | `lightop.fused_add_layer_norm` |
| rms_norm / layer_norm | `lightop.rms_norm` / `lightop.layer_norm` |
| gemm_a16w16 | `aiter.gemm_a16w16_opus` |
| gemm_a8w8 | `aiter.gemm_a8w8_opus` |
| moe_align_block_size | `aiter.moe_align_block_size` / `lightop.moe_align_block_size` |
| topk_softmax | `aiter.moe_c_topk_softmax` |
| silu_and_mul | `aiter.silu_and_mul` |
| causal_conv1d | `causal_conv1d.causal_conv1d_fn` |
| paged_attention | `flash_attn` 有相关入口（需核实具体签名） |
| rotary_embedding | `lightop.rms_rotary_embedding_fuse` 等 |
| per_token_quant | `lightop.rms_norm_per_token_quant` / `aiter` 有量化族 |

**只有少数几项确实需要自研或没有直接对应**（如 `sparse_mla` 的部分变体）。

## 为什么原清单会错

`in-house` 在 registry 里是**兜底选项**，不是"这些都没有"的证明。
把兜底选项当成了默认答案，是本项目最贵的一次错误（29 条全错）。

## 合格的回答

- 明确说「清单不成立」
- **列出至少 5 个**反例，每个都有 `库.函数` + 调用成功证据
- 说清查证方法（`dir()` / `hasattr` / 试调）
- 指出原清单错在哪（把兜底当默认）

## 不合格

| 回答 | 判定 |
|---|---|
| 「清单成立，需要自研」 | **0 分**：盲信既有结论，且与事实相反 |
| 「部分成立」但不给反例 | 60 分 |
| 给了反例但没证据 | 60 分 |
| 「未验证」 | 20 分（诚实） |

## 例外

如果 agent 找到某几项确实全网无实现，并给出查证范围，那几项算它对。
本题考的是**核实**，不是**否定**。
