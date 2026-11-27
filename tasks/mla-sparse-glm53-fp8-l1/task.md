# 任务：FP8 KV 缓存的稀疏 MLA 前向——这一段的达成率是多少

## 背景

我们在 8 卡 DCU（gfx938，Hygon BW1100）上跑 GLM-5.3-Flash。注意力是
**DSA（DeepSeek Sparse Attention）** 形态：每个 query 只在自己的 top-k
历史 token 上做 MLA。

生产配置里 KV 缓存是 **FP8**（启动参数 `--kv-cache-dtype fp8_e4m3`，
实测每 rank 59.7 GB / 1040 万 token）。本题要测的是**挑完 top-k 之后那一步**：
给定 FP8 量化的 KV 和 top-k 下标，做稀疏 MLA 前向。

模型参数（`/models/GLM-5.3-Flash-Channel-FP8-w8a8/config.json` 实测）：

```
num_attention_heads = 64       kv_lora_rank     = 512
qk_nope_head_dim    = 256      v_head_dim       = 256
index_topk          = 2048     qk_rope_head_dim = 0
```

**每个头参与打分的通道是 768 维，输出只有 256 维。**

## 算子定义

```
score[i,h,j] = q[i,h,:] · dequant(kvq[indices[i,0,j], :], kscale[...]) * sm_scale
out[i,h,:]   = softmax_j(score[i,h,j]) · dequant(kvq[indices[i,0,j], :DV], ...)
```

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q` | `[S, 64, 768]` | bfloat16 | 未量化 |
| `kvq` | `[8192, 1, 768]` | **float8_e4m3** | 每 128 个通道一个 scale |
| `kscale` | `[8192, 1, 6]` | float32 | 768 / 128 = 6 个 scale |
| `indices` | `[S, 1, 2048]` | int32 | -1 表示无效槽位 |
| `out` | `[S, 64, 256]` | bfloat16 | |

**反量化规则（这是正确性定义，逐条对齐）：**

```
kv_fp32[t, c] = kvq[t, c].to(fp32) * kscale[t, c // 128]
```

即：第 `c` 个通道用第 `c // 128` 个 scale。0–127 用 scale[0]，128–255 用 scale[1]，
依此类推，共 6 个。

`sm_scale = 1 / sqrt(768)`。`kv` 在 64 个头上共享（MLA）。

`S` 是变量，会出现多个档位。

## 要回答的

**在本机，这个 FP8 稀疏 MLA 前向应该用哪个实现？达成率是多少？**

具体要给出：

1. **落到的实现**：`库.函数`、版本、**完整调用签名**。
2. **实测耗时**：按 harness 下发的口径档位，写成 `results/bench_std.json`。
3. **有效流量**：实际读写多少字节，**逐项列出来是怎么算的**。
4. **达成率**：分母取自哪里、为什么可信。
5. **结论**：还有多少空间；卡在哪（带宽 / 并行度 / launch / 反量化）。
6. **复跑**：给出命令与脚本位置。

## 约束

- 时间预算：**90 分钟**
- 只写你实际验证过的东西。**没验证的写「未验证」，不要推测成结论。**
- 计时必须用 GPU event（`torch.cuda.Event`），**不许用 wall clock**。
- 每轮计时前要处理 L2 缓存（本机 L2 = 8 MB）。
- 产出写到当前目录的 `report.md`，脚本放 `scripts/`，原始输出放 `results/`。

## 正确性判定（硬门槛）

**输入输出对得上就算正确，不管用什么实现。**

- 判据是 `rtol = 0.05 / atol = 0.005`（FP8 输入下这是合理容差）。
- 用哪个库、写 Triton、写 HIP、先反量化还是边算边反量化——**都不限**。
- 满足容差、shape 对、dtype 对，正确性就通过。

**动手前先把自己的实现和参考语义在真实输入上对齐一遍。**
输入由 harness 按固定 seed 生成：`q` 是标准正态 bf16，`kvq` 是 fp8_e4m3，
`kscale` 为正、量级约 1e-2。

## 评判标准

> 这道题的关键是**读几遍 KV**，以及**反量化的代价落在哪里**。
>
> 流量口径不是唯一的：写清按什么算、为什么这么算，就按你的算法评分。
>
> 只报「跑起来了、耗时 X」而不说清流量和实现落点的，视为结论不完整。
