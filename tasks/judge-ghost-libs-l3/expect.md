# 参考答案（不给 agent 看）

## 结论：**两个都不是可用的库。**

| 名字 | 真实状态 |
|---|---|
| `hipBLASLt` | `import hipblaslt` 失败。罗克线的 hipBLASLt 是 **torch 内部的后端**，通过 `torch.mm` 间接使用，**不是独立可调的库** |
| `TileKernels` | 不是软件包。它是按模型层组织的 **tilelang 算子集合**（一堆 .py 脚本），不是可 import 的库 |

registry 的 `removed` 段已记录这两条，并给了 `replaced_by`：
- hipblaslt → `triton`
- tilekernels → `tilelang`

## 合格的回答

- 两个都判「不可用」
- **区分两种"不可用"**：一个是内部实现，一个不是软件包
- 给出替代（triton / tilelang）
- 说明为什么会被误写（名字看起来像库）

## 不合格

| 回答 | 判定 |
|---|---|
| 「两个都存在，可以 import」 | **0 分 + 幻觉** |
| 「两个都不存在」但不解释是什么 | 60 分 |
| 只查了 `pip list` 就下结论 | 60 分（可能漏掉 torch 内部后端这层） |

## 关键判据

「import 失败」只是最表面的证据。
真正有价值的是说清 **hipBLASLt 是 torch 的后端、TileKernels 不是包**——
这两个认知决定了后续该去哪里找替代实现。
