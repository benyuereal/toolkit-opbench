# 参考答案（不给 agent 看）

## 语义

```
kv_fp32[t, c] = kvq[t, c].to(fp32) * kscale[t, c // 128]

score[i,h,j] = q[i,h,:] · kv_fp32[indices[i,0,j], :] * sm_scale
out[i,h,:]   = softmax_j(score[i,h,j]) · kv_fp32[indices[i,0,j], :DV]
```

D=768（512 latent + 256 nope），DV=256，H=64，TOPK=2048，SKV=8192。
`sm_scale = 1/sqrt(768)`。

## 流量

| 项 | 字节 | S=128 | S=2048 |
|---|---|---|---|
| 读 `q`（bf16） | `S*H*D*2` | 12.58 MB | 201.33 MB |
| 读 `kvq`（**fp8，1 字节**） | `S*TOPK*D` | 201.33 MB | 3221.23 MB |
| 读 `kscale`（每 128 通道 1 个 fp32） | `S*TOPK*QTILE*4` | 6.29 MB | 100.66 MB |
| 读 `indices` | `S*TOPK*4` | 1.05 MB | 16.78 MB |
| 写 `out`（DV=256） | `S*H*DV*2` | 4.19 MB | 67.11 MB |
| **合计** | | **225.4 MB** | **3607.1 MB** |

**FP8 的收益在 kv 项**：从 bf16 的 `S*TOPK*D*2` 变成 `S*TOPK*D`，正好减半。
但 scale 项额外要读 `S*TOPK*24` 字节，占 kv 项的 3.1%——省 50% 之后
还要还回去 3.1%，净省 46%。

对比 bf16 版（`mla-sparse-glm53-l1`）：S=128 从 420.5 MB 降到 225.4 MB。

**分母底线**：`peak_achievable:read = 1870 GB/s`（2026-11-26 由 1965 修正）。
S=128 带宽地板 0.121 ms，S=2048 地板 1.929 ms。两个档位都是**带宽地板支配**
（S=128 也是 launch 地板 0.043 ms 的 2.8 倍）。

## 基线与实测（`baselines/mla-sparse-glm53-fp8-l1.json`）

| S | 流量 | 地板 | 实测 `t_b` | 达成率 | cv |
|---|---|---|---|---|---|
| 128 | 225.4 MB | 0.121 ms | **1.598 ms** | **7.5%** | 0.0011 |
| 2048 | 3607.1 MB | 1.929 ms | **22.129 ms** | **8.7%** | 0.0004 |

`t_b = 5.9474 ms`（几何平均），`t_sol = 0.4822 ms`，**上限 12.33x**。
sclk 1200 MHz（档位 5）。cv 极低说明测量干净，这个基线可信。

**这是全仓上限最大的一道题**（对照：conv1d 8.20x、bf16 版 sparse MLA 6.18x）。

## 基线为什么这么慢：它是降级路径

本机**没有**能吃 GLM 形状的 FP8 稀疏 MLA kernel（2026-11-26 逐个试过）：

| 入口 | 结果 |
|---|---|
| `flash_mla.flash_mla_sparse_fwd` | ❌ `Invalid d_qk: 768` |
| `flash_mla.flash_mla_with_kvcache_fp8` | ❌ 只支持 d_qk∈{512,576}、d_v=512 |
| `aiter.ops.tilelang.sparse_mla_fwd` | ❌ 传 fp8 报 `ufunc_add_CUDA not implemented for Float8_e4m3fn` |
| `sglang.kernels.ops.attention.dsa.triton_sparse_mla` | ✅ 但只吃 bf16 |

所以基线是生产里真实的降级做法：**整块反量化成 bf16，再走已验证的 bf16 路径**：

```python
s = kscale.repeat_interleave(128, dim=-1)
kvd = (kvq.float() * s).to(bfloat16)          # 整块反量化：6.3 MB → 12.6 MB
out = triton_sparse_mla_fwd(q_nope, q_rope, kvd, indices, sm_scale, 256)
```

这个降级本身就是要测的东西——它多了一趟全量反量化，而反量化只需要覆盖
被 gather 的那部分。

## 优化空间在哪（题面不说）

三件事，按收益排序：

**① 反量化与 gather 融合。** 基线把 8192 行全反量化（覆盖 4096 个被选中的
token 里可能只用到一小部分重叠），再 gather。正确做法是 gather 之后就地反量化，
反量化的字节量与 gather 量一致。但要注意：整块反量化只有 6.3 MB，
而 gather 量在 S=2048 时是 3.2 GB——**所以这一条在小 S 才有意义**，
大 S 时反量化开销可以忽略。别把它当成主要收益。

**② 真正的瓶颈是 bf16 kernel 本身。** 22 ms 对 1.93 ms 地板，差 11 倍。
原因同 bf16 版：`triton_sparse_mla_fwd` 只按 `seq` 一维开 program
（`_sparse_mla_fwd_kernel[(seq,)]`），每个 program 顺序走完整个 topk，
softmax 与 P·V 串行。**空间在并行度（split-KV），不在反量化。**

**③ 直接在 FP8 上做 QK^T。** 理论上可以省掉反量化，用 fp8 MMA。
但本机没有这样的 kernel，要自己写；而且 V 侧求和仍在 fp32。

> 注意：①看起来最直观（"我一眼就看出整块反量化浪费了"），但它不是主要空间。
> 只盯着这一条、优化到极致也只能拿到个位数百分比。真正 10 倍的空间在 ②。

## 三个语义坑

### 1. scale 粒度是 128 个通道

`kscale[t, c // 128]`，768 维共 6 个。用成「整个 token 一个 scale」会让
反量化值偏 3–6%，`rel_l2` 到 30% 量级（实测：0.3 vs 0.003），判据会抓住。
用成「逐通道 768 个」形状直接对不上。

### 2. 哨兵 `-1` 与 `>= SKV` 都无效

softmax 前掩成 `-inf`，**不能当索引 0 去 gather**。当 0 会让 `rel_l2` 到 50%
量级（实测 max_abs 2.9），比正确值大三个数量级。整行全哨兵时输出 0。

### 3. 输出只有 DV=256 维

nope 后缀 `[256:]` 只参与打分，不进输出。按 768 算会高估输出项 3 倍。

## 合格答案

- 发现本机没有 GLM 形状的 FP8 kernel，并对每条路给出具体报错
- 落到降级路径（先反量化再走 bf16 kernel），说清这是降级
- 实测出 7.5%–8.7% 的达成率，指出 11 倍空间
- **能分清「反量化开销」和「kernel 并行度」哪个是大头**
- 说清 `-1` 哨兵、scale 粒度 128、输出 256 维

## 不合格

| 回答 | 判定 |
|---|---|
| 「用 flash_mla_with_kvcache_fp8，跑通了」 | **0 分**：GLM 形状它直接拒绝 |
| 说「FP8 省了一半带宽所以应该快一倍」 | 60 分：没测，流量账也算错（漏了 scale 项） |
| 认为主要空间在「整块反量化太浪费」 | 60 分：那是小 S 的事，大 S 差 11 倍是 kernel 本身 |
| 流量把 kv 按 bf16 的 `D*2` 算 | 60 分：FP8 的意义就没了 |
| 输出按 768 维算 | 60 分：高估 3 倍 |
| 把 -1 当索引 0 | **0 分**：数值全错 |
| 「未验证」 | 20 分（诚实） |

## 判据要点

与 bf16 版相比，这题多一层：**FP8 的"省带宽"是账面收益，实际能不能兑现
取决于 kernel 有没有用上**。

- 裸 agent：很可能直接说"FP8 省一半带宽"，或者撞 `Invalid d_qk: 768` 后
  不知道该退到哪
- 有 skill：知道本机 DSA 的落点、形状限制，也更容易分清
  "反量化开销" 与 "kernel 并行度" 哪个才是主要瓶颈
