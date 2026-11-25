# 参考答案（不给 agent 看）

## 语义

MoE 计算前，token 必须按专家重排成**块对齐**的布局，下游 grouped GEMM
才能按固定 tile 高度切分。

```
输入   topk_ids[M, K]          每个 token 选中的 K 个专家 id
         E=64, K=8, ALIGN=128
输出
  sorted_ids[CAP]        重排后的 token 下标（token*K + k）
  expert_ids[BLOCKS]     第 b 个 128-slot 块属于哪个专家
  num_post_pad[1]        前多少个 sorted_ids 有效
  CAP = M*K + E*ALIGN,  BLOCKS = CAP // ALIGN
```

## 实现栏位上的坑

`lightop.moe_align_block_size(topk_ids, block_size, num_experts)` ——
**`block_size` 在前，`num_experts` 在后**，与直觉相反。

实测按 `(topk_ids, num_experts, block_size)` 传会抛
`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。
那个报错和参数顺序毫无关系，很容易被误导去查 dtype。

## 流量

| 项 | 字节 | M=8192 |
|---|---|---|
| 读 `topk_ids` | `M*K*4` | 0.262 MB |
| 写 `sorted_ids` | `CAP*4` | 0.295 MB |
| 写 `expert_ids` | `BLOCKS*4` | 0.002 MB |
| 写 `num_post_pad` | 4 | 0.000004 MB |
| **合计** | | **0.559 MB** |

## 这是 launch 判据题，不是带宽题

| | 值 |
|---|---|
| 数据量（M=8192） | 0.559 MB |
| 带宽地板 @1350 | **0.0004 ms** |
| launch 地板 | **0.043 ms**（constants 的 launch.floor_us） |
| 实测基线 | 0.0931 ms |
| 上限 | **2.17×**（取 max(带宽, launch) = launch） |

**如果按带宽算，会得出"上限 341×"——那是个不存在的问题。**
没有任何实现能在一个 launch 里做到比 launch 更快。拿这个数驱动优化，
会让人去改 kernel 内部，而真正的方向是减少 launch 次数或换算法结构。

所以 `t_sol` 取 `max(带宽地板, launch 地板)`，与
`bench/score/perf.py:roofline_floor()` 用同一口径。

## 正确性判据（本题用自定义 hook）

**不能用逐位比**，两个原因：

1. **`sorted_ids` 的顺序不唯一**。同一专家内部，token 先后由实现决定。
   `argsort(stable=True)` 与库的计数排序给同一集合、不同顺序。
2. **padding 用 0 填充，尾部是脏数据**。实测 lightop 的 `sorted_ids[:npd]`
   里 `0` 出现 39 次（1 次真实 + 38 次 padding）——所以"有效段里 0 重复"是
   **正常**的，不能据此判错。

判据：

| 检查 | 方式 |
|---|---|
| `num_post_pad` | 必须相等 |
| `sorted_ids[:npd]` | 去掉 padding 的 0 后，**非零集合**相等 |
| `expert_ids[:ceil(npd/ALIGN)]` | **逐位相等** |

不逐段数"非零 token 个数"：padding 的 0 与 token 0 无法区分，
实测这条会把正确的 lightop 判错（段内计数 577 vs 640）。

## 合格答案

### ① 最好：认出这是 launch 主导，给出正确归因

测出 93µs，算出带宽地板 0.4µs，指出差 200 倍 → 瓶颈不在带宽 →
结论"再优化 kernel 内部没用，该看能不能合并到上游/下游 kernel"。

### ② 次好：换更快的库实现

如果 `aiter.moe_align_block_size` 或 `moe_c_sgl_moe_align_block_size`
比 lightop 快，如实报告并给数字。

### ③ 可接受：自己写，但要说清为什么值得

只有当它能显著减少 launch 次数时才有意义。

## 不合格

| 回答 | 判定 |
|---|---|
| 「达成率 2.5%，还有 40 倍空间，应该优化 kernel 内部」 | **60 分**：把 launch 开销当成 kernel 效率问题 |
| 流量按 `M*K*4` 只算读 | 60 分：漏掉约 50% 的写 |
| 逐位比 `sorted_ids`（把正确的库实现判成错） | 60 分：判据错 |
| 按 `(topk_ids, num_experts, block_size)` 传参 | 0 分：跑不起来 |
| 「未验证」 | 20 分（诚实） |
