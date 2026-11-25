# 参考答案（不给 agent 看）

## 当前已知结论（截至 2026-11-25）

**落点：`sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd`**
（sglang 0.5.18 / DTK-26.04-rc4，本机）。这是 sglang 自带的 DSA 稀疏 MLA
Triton kernel，也是本机 GLM 服务的 DSA 路径上游。

签名（实测，`inspect.getsource`）：
```
triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=256) -> [1, S, H, d_v] bf16
    q_nope:  [S, H, d_v]       q 的前 d_v 维（bf16）
    q_rope:  [S, H, D-d_v]     q 的后 D-d_v 维（本题 768-256=512）
    kv:      [SKV, 1, D]       bf16
    indices: [S, 1, TOPK]     int32，-1 为无效槽位
```

## 本题的核心：flash_mla 吃不下这个形状

这是本题与 `mla-sparse-l1`（576/512）最本质的区别。**GLM 是 768/256**。

实测（2026-11-25，全部真跑过）：

| 入口 | 形状 | 结果 |
|---|---|---|
| `flash_mla.flash_mla_sparse_fwd` | d_qk=768, d_v=256 | ❌ `RuntimeError: Invalid d_qk: 768` |
| `flash_mla.flash_mla_sparse_fwd` | d_qk=576, d_v=256 | ❌ `RuntimeError: Invalid d_v256`（d_v 只能 512） |
| `flash_mla.flash_mla_with_kvcache(indices=...)` | d_qk=768 | ❌ `DSA BF16 sparse decode only supports d_qk == 512 for now` |
| `aiter.ops.tilelang.sparse_mla_fwd` | dim=256/tail=512 | ⚠️ 能编译，但**带 causal 掩码**：随机下标下 max_abs 0.2–0.57（错） |
| `aiter.ops.tilelang.sparse_mla_fwd` | causal-consistent 下标 | ✅ max_abs 0.016（但本题 indices 是随机的，非因果） |
| `sglang.kernels.ops.attention.dsa.triton_sparse_mla` | 768/256 | ✅ max_abs 2e-3（随机下标） |
| `sglang.kernels.ops.attention.dsa.tilelang_kernel.tilelang_sparse_fwd` | 768/256 | ✅ 数值对，但**极慢**（S=128 12.7 ms，S=2048 192 ms） |
| `aiter.ops.tilelang.sparse_mla_fwd`（旧签名） | d_qk=768 | ❌ `AssertionError: you should assign dim otherwise`（断言写死 576） |

`flash_mla` 的 Python 层没有任何报错提示"支持什么形状"，`Invalid d_qk: 768`
是从 C++ 里抛的——不看源码根本不知道它只认 512/576。这是本题第一个坑。

## 三个语义坑

### 1. 输出只有 DV=256 维，不是 D=768 维

`q`/`kv` 是 768 维（512 latent + 256 nope），但**输出只取前 256 维**
（对应 `v_head_dim=256`）。nope 后缀 `[256:]` 只参与 `q·kv` 打分，不参与输出。

流量过估的直接后果：把输出按 768 算会高估 `S*H*DV*2` 项 3 倍。
S=128 时输出只有 4.19 MB、S=2048 时 67 MB——在小 S 下这一项占比不大，
但「输出该按多宽算」说不清就是没对齐语义。

### 2. kv 流量按「被 gather 的量」算，不是按原始缓冲算

`kv` 本体只有 `SKV*D*2 = 12.6 MB`（8192 × 768 × 2）。
但每个 query 都要 gather 自己 topk 个 token 重读一遍：
每 token `TOPK*D*2 = 3.15 MB`。S=128 时 402.7 MB。

这是本题**最大的流量项**（S=128 占总流量的 96%，S=2048 占 96%）。
把 kv 按 12.6 MB 算会得出「带宽地板亚毫秒」，与实际耗时差两个数量级。

### 3. 哨兵 `-1` 与 `>= SKV` 都是无效槽位

`indices == -1` 表示没选到 token，softmax 前必须掩成 `-inf`，
**不能当索引 0 去 gather**。当 0 会静默算错：

实测 `sentinel-as-index0` vs 正确参考（30% 哨兵，S=128）：
204 万 / 210 万元素超差，max_abs 2.9。容差 rtol=0.05/atol=0.005 下
**不会**被放过。整行全哨兵时输出 0。

## 流量公式（definition.dcu.work）

```
S*H*D*2      读 q
S*TOPK*D*2   读 kv（每个 query gather topk 个，MLA 下 kv 分头共享）
S*TOPK*4     读 int32 indices
S*H*DV*2     写 out（DV=256）
```

S=128：q=12.58 MB + kv=402.65 MB + ind=1.05 MB + out=4.19 MB = 420.48 MB
S=2048：q=201.33 MB + kv=6442.45 MB + ind=16.78 MB + out=67.11 MB = 6727.66 MB

分母 `peak_achievable:read = 1965 GB/s` → 带宽地板
S=128 = 0.214 ms，S=2048 = 3.424 ms。T_sol = max(带宽地板, launch 地板 0.043 ms)，
所以两个档位都是**带宽地板支配**（S=128 也远高于 launch 地板）。

## 基线数字怎么读（见 `baselines/mla-sparse-glm53-l1.json`）

本题是**带宽题**（不是 launch 题）——S=128 的地板 0.214 ms 是 launch 地板
0.043 ms 的 5 倍。但实测 sglang Triton kernel 在 S=128 约 1.4 ms、
S=2048 约 22 ms，离地板很远：

| S | 流量 (MB) | 带宽地板 (ms) | 实测 (ms) | roofline 达成率 |
|---|---|---|---|---|
| 128 | 420.5 | 0.214 | ~1.40 | ~15% |
| 2048 | 6727.7 | 3.424 | ~22.0 | ~16% |

**注意**：这是 2026-11-25 在共享机上测的（卡 0，测量窗口里同机有其他
pytest/opcheck 任务），绝对值可能被邻居带偏 10–50%。请以
`baselines/mla-sparse-glm53-l1.json` 里记录的 t_b 为准，并参照 sclk 注记。

达成率只有 ~15% 说明**这个 kernel 离极限很远**——S=2048 时 22 ms 远高于
6.4 GB / 1.965 TB/s = 3.4 ms。原因不在内存带宽，而在：
kernel 只按 `seq` 一维开 program（`_sparse_mla_fwd_kernel[(seq,)]`），
每个 program 顺序走完整个 topk，refill 循环里 softmax 与 P·V 串行。
真正的优化空间在并行度（split-KV）与 P 的 fp8 化——但这两点
**不该写进题面**（那是解法）。

## 合格的回答

- 落到 sglang 的 `triton_sparse_mla_fwd`（或其上游同款 DSA kernel），写出签名与 q 拆两段
- **实测出 flash_mla 拒绝 768/256**（这条最能区分"真跑了"和"看了文档"）
- 说清 `-1` 哨兵语义、输出只取前 256 维
- 流量按 gather 量算 kv、按 256 维算输出
- 至少两个 S 档位，给出带宽地板与达成率

## 不合格的回答

| 回答 | 判定 |
|---|---|
| 「用 flash_mla_sparse_fwd，跑通了」 | **0 分**：768 形状它直接拒绝 |
| 用 torch 手写 sparse attention 当"优化实现" | 60 分：能跑但没落到本机 kernel |
| 流量把 kv 按原始 12.6 MB 算 | 40 分：地板差两个数量级，达成率无意义 |
| 输出按 768 维算 | 60 分：对输出项高估 3 倍 |
| 把 -1 当索引 0 | **0 分**：数值全错（204/210 万元素超差） |
| aiter tilelang 当落点但不发现它的 causal 语义差 | 50 分：随机下标下是错的结果 |
| 「未验证」 | 20 分（诚实） |

## 判据要点

本题三个坑里有两个（flash_mla 的形状拒绝、aiter tilelang 的 causal 语义）
**不可能从签名或文档看出来**，只可能来自：真跑过、或读过 skill 的 failure-modes。

- 裸 agent：极可能直接调 `flash_mla_sparse_fwd`，撞 `Invalid d_qk: 768` 后
  不知道该退到哪；或者退到 `tilelang_sparse_fwd`（能跑但慢 10 倍，
  会把达成率算成 ~1.5%）
- 有 skill：知道本机 DSA 的可用落点与形状限制
