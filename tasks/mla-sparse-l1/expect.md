# 参考答案（不给 agent 看）

## 当前已知结论（截至 2026-11-24）

**落点：`flash_mla.flash_mla_sparse_fwd`**（flash_mla 1.2.0）。

签名（实测）：
```
flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512,
                     attn_sink=None, topk_length=None)
    -> (out, lse, max_logits)
q: [s_q, h_q, d_qk] bf16;  kv: [s_kv, h_kv, d_qk] bf16
indices: [s_q, h_kv, topk] int32;  d_v 只能 512
```

## 三个坑（本题的核心）

### 1. 形状约束：topk 必须是 128 的倍数且 ≥128

实测（这是**跑出来的**，不是文档写的）：

| topk | 结果 |
|---|---|
| 64 | ❌ `Assertion params.topk % (2*B_TOPK) == 0 failed` |
| 128 / 256 / 512 / 1024 | ✅ OK |

只接受 ≥128 且是 128 倍数的 topk。**这条只有实际跑才会发现**——
不查就去写调用示例，会写出一个跑不通的版本。

### 2. 哨兵语义：`-1` 和 `>= SKV` 都是无效

`indices == -1` 或 `>= s_kv` 表示该槽位没选中 token，softmax 前必须掩成
`-inf`，**不能当成索引 0**。把它当 0 会静默算错（输出仍然有限、看起来正常）。

生产里 padding 比例不低（短上下文时大量槽位是 -1），所以
`effective_topk = mean(indices >= 0) * topk` 要单独统计。

### 3. 输出只有 512 维，rope tail 不输出

`q`/`kv` 是 576 维（512 latent + 64 rope tail）。
rope tail **只参与打分，不参与输出**。

流量口径：
```
读 q:   S * H * 576 * 2
读 kv:  S * TOPK * 576 * 2      ← 每个 query 稀疏 gather topk 个
写 out: S * H * 512 * 2         ← 注意是 512 不是 576
```

把输出按 576 算会高估流量 12.5%，达成率跟着错。

## sm_scale 口径

`sm_scale = 1/sqrt(d_qk) = 1/sqrt(576)`（本档位 d_qk=576）。
**不是** 1/sqrt(640) —— 640 是 latent+rope 之外的另一个口径
（`decode-glm53` 那种 d_qk=512 的档位也容易混）。

## 合格的回答

- 落到 `flash_mla.flash_mla_sparse_fwd`，写出完整签名
- **实测出 topk 必须是 128 倍数**（这条最能区分"真跑了"和"看了文档"）
- 说清 -1 哨兵语义
- 流量按 512 维输出算，不是 576
- 至少两个 S 档位

## 不合格的回答

| 回答 | 判定 |
|---|---|
| 「用 flash_mla，跑通了」但 topk=64 的示例 | **0 分**：示例跑不通，且没发现约束 |
| 用 torch 手写 sparse attention 当"优化实现" | 60 分：能跑但没落到本机 kernel |
| 流量按 576 维输出算 | 60 分：达成率错 12.5% |
| 把 -1 当索引 0 | **0 分**：静默错误 |
| 「未验证」 | 20 分（诚实） |

## 判据要点

本题**不是**考"调通一个 API"。三个坑里有两个（形状约束、哨兵语义）
**不可能从签名看出来**，只可能来自：跑过、或读过 skill 的 failure-modes。

- 裸 agent：可能踩坑（写个 topk=64 的示例就崩）
- 有 skill：`op-sparse-mla` 的 SKILL.md 明确写了 -1 哨兵与形状档位
