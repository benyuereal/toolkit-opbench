# 参考答案（不给 agent）

> 2026-11-25 于卡 4（sclk 1350 MHz）实测。

## 1. 本机有哪些现成 per-token 量化实现

| 入口 | 类型 | 语义 | 稳定性 |
|---|---|---|---|
| `lightop.quant.per_token_quant_fp8(x, dtype=fp8_e4m3, out_q=None, out_scale=None)` | HIP kernel | 与定义完全一致 | 稳定 |
| `aiter.per_token_quant_hip(x, scale=None, quant_dtype=fp8_e4m3, ...)` | HIP kernel | 一致 | **偶发 NotImplementedError** |
| `aiter.per_token_quant_triton(x, scale=None, quant_dtype=...)` | Triton | 一致（零行 scale 是 2.2e-13 而非 1.0） | 稳定但慢 |
| `aiter.dynamic_per_token_scaled_quant(out, x, scales)` | HIP kernel（底层 C++ 入口） | 一致，需预分配 | 稳定 |
| `aiter.pertoken_quant(x, ...)` | **纯 torch** | 用 `.round()`，fp8 上一致 | 不是 kernel，不能当基线 |
| `lightop.rms_norm_per_token_quant` / `rms_norm_per_token_fp8_quant` | HIP kernel | **多一层 rmsnorm**，要 weight | 语义不同，不是纯量化 |
| `aiter.per_token_quant_hip` int8 分支 | HIP kernel | **用截断**（trunc），范围 [-127,127]，不取 -128 | — |

`lightop.per_token_quant_fp8` 只在 **子模块** `lightop.quant` 下，不在顶层。

### 已知环境缺陷（**不要写进 task.md**）

`aiter.per_token_quant_hip`（fp8 分支）偶发：

```
NotImplementedError: "min_all_cuda" not implemented for 'Float8_e4m3fn'
```

来源是 `aiter/jit/utils/torch_guard.py` 的 `wrapper_custom`——它把调用转成
`torch.ops.aiter.<name>`；FP8 的 fake/abstract 路径在 `min_all` 上炸。
实测 200 次连调 0 失败，但换 seed/换张量后会零星触发，触发一次就中断整轮计时。
**所以基线用 lightop，不用 aiter。**

## 2. 基线（卡 4，warmup=50，iters=200，flush 16 MB，机器空闲时）

| uuid | M | median | cv |
|---|---|---|---|
| pub-2048 | 2048 | 0.0618 ms | 0.074 |
| pub-8192 | 8192 | 0.0803 ms | 0.010 |

`t_b`（跨形状几何平均）= **0.0705 ms**
`t_sol` = max(带宽地板 0.0566, launch 地板 0.043) = **0.0566 ms**（带宽支配）
上限倍数 `t_b / t_sol` = **1.24×**（跨多次复跑 69.5–76.7 µs，上限在 1.23–1.35× 之间）

注意：`t_sol` 在**每一个 public 档位上**分别是 M=2048 → 43 µs（launch 支配）、
M=8192 → 74.6 µs（带宽支配）；几何平均后落在 56.6 µs。也就是说
"上限 1.24×" 这个数本身是两种瓶颈混在一起的产物，**不能当成单一瓶颈读**。

### 这是带宽题还是 launch 题：**随 M 换档，不是单一一类**

用**预分配**版本（排除分配噪声）测的扫描，最能说明归因：

| M | 实测 | 带宽地板 | 倍数 |
|---|---|---|---|
| 512 | 37.8 µs | 4.7 µs | 8.1× |
| 1024 | 38.4 µs | 9.3 µs | 4.1× |
| 2048 | 37.1 µs | 18.6 µs | 2.0× |
| 8192 | 80.5 µs | 74.6 µs | 1.08× |
| 13312 | 125.9 µs | 121.2 µs | 1.04× |
| 16384 | 153.4 µs | 149.2 µs | 1.03× |
| 32768 | 299.1 µs | 298.4 µs | 1.00× |

**小 M 段被固定开销吃掉（~37 µs 基本不随 M 变），大 M 段贴着带宽
（M≥8192 达成率 92–100%）。** 分界点大约在 M≈4096。

关键归因数字：M 从 512 涨到 32768（64×），耗时从 37.8 涨到 299.1 µs（7.9×）。
若纯带宽，耗时应与 M 成正比；实测远低于此比例，说明有一部分是**与 M 无关的
固定成本**。

### 还有没有空间：**大 M 有 ~15%，小 M 是固定开销**

同一流量比（读 2 字节 bf16 / 写 1 字节 fp8 + scale）的**纯 cast**（`x.to(fp8)`，
不做 per-token，只当"同流量下的实现上限"）实测：

| M | 纯 cast | 纯 cast GB/s | per-token kernel | kernel GB/s | kernel/cast |
|---|---|---|---|---|---|
| 8192 | 78.5 µs | 1283 | 80.3 µs | 1254 | 1.02 |
| 13312 | 109.3 µs | 1497 | 126.0 µs | 1299 | 1.15 |
| 16384 | 132.5 µs | 1520 | 153.1 µs | 1315 | 1.16 |
| 32768 | 256.0 µs | 1573 | 298.4 µs | 1350 | 1.17 |

**结论**：
- 大 M（≥13312）：per-token kernel 比同流量纯 cast 慢 **15–17%**。这部分是
  kernel 内部（多一趟 amax 归约、或未做向量化）的真实差距，可以通过自己写
  kernel（一趟遍历同时求 amax + 量化）拿回。
- 小 M（≤2048）：瓶颈是每次调用的固定开销（launch + 输出分配），
  kernel 内部怎么改都没用，**要改的是缓冲区复用 / 减少 launch**。
- 注意分母口径：`peak_achievable:copy`=1350 GB/s 是 **R:W=1:1** 的档。
  本算子 R:W=2:1，constants 里对应的 2:1 档是 1730 GB/s —— 所以
  "1350 GB/s"对本算子其实是偏保守的分母（kernel 到 1350 已是 2:1 档的 78%）。
  但 1730 只存在于点分路径且 `unit=None`，不符合 `baseline.build()` 的单位校验，
  所以基线仍用 `peak_achievable:copy`。

### 分配开销（这是 M=2048 档的主要固定成本）

`lightop.per_token_quant_fp8` 不传 out_q/out_scale 时每次调用会
`torch.empty_like` 两个输出。CPU 侧配对分配实测 7.6 µs/call，但 GPU 上
计入 event 后 M=2048 差 **~20 µs**（alloc 56–58 µs vs prealloc 35–37 µs），
M=8192 差 ~10 µs。**agent 若预分配/复用输出缓冲区，中小 M 有实质收益。**

## 2b. 基线数字的噪声（如实记录）

这台机器是共享的：`harness.verify` 或别人的作业一开，八张卡全被拉满。
同一份代码在不同时刻测出的 M=2048 档在 57–71 µs 之间漂（cv 最高 0.28）。
本基线取的是**机器空闲窗口**（`hy-smi -u` 全 0、无 `harness.verify`）下
连续三次里居中的那次；连续多次的 `t_b` 是 69.5 / 69.6 / 70.5 / 72.2 / 73.8 / 74.2 / 76.7 µs，
离散度 ~10%。复跑必须在同样条件下做，否则比的是邻居负载。

## 3. 正确性判据依据

实测正确实现（vs 反量化后的原始 x，M∈{2048..16384} × {normal, ×1e4, ×1e-4, 交替符号, 全 1}）：

| 实现 | rel_l2 | max\|d-x\|/rowmax |
|---|---|---|
| lightop / aiter_hip / aiter_triton | 0.0265 | 0.0355 |
| 截断代替 RNE | 0.0271 | 0.0379 |

阈值取 **rel_l2 ≤ 0.05（1.9× 余量）、逐元素/rowmax ≤ 0.10（2.8× 余量）**，
scale 契约 rtol ≤ 1e-4（正确实现 0.0；bf16 scale 0.4%；qmax 写 127 时 3.5×）。

### 抓得住的错

| 错误 | 被哪条抓 |
|---|---|
| per-tensor 而非 per-token | scale 检查（各行 scale 应不同） |
| qmax 写成 127 | scale 检查（3.5×） |
| scale 存 bf16 | scale 检查（0.4% > 1e-4）；反量化 rel_l2 仍 0.0265 |
| 输出全 0 / 原样拷贝 | 反量化 rel_l2 = 1.0 |
| 输出形状/dtype 错、NaN | 形状/dtype/有限性检查 |

## 4. 最容易被答错的点（题面**没有**提示，故意的）

1. **只看一个 M 就下结论**。M=8192 单点结论是"92% 达成率，没空间"，
   而 M=2048 单点结论是"还有 1.4×"。两者都对，合起来才是真相：
   固定开销在小 M 显著。题面只说"多测几个 M"级别的提示，没点破比例。
2. **把分配开销算成 kernel 开销**。不预分配 vs 预分配在 M=2048 差 25 µs。
3. **per-tensor vs per-token**：`aiter.per_tensor_quant` 就在旁边，
   签名几乎一样，选错不会报错、只在 scale 检查上暴露。
4. **scale dtype**：`scale` 用 bf16 存不会报错，反量化 rel_l2 也不变，
   只有 scale 契约（1e-4）能抓住。
5. **round 方向**：fp8 走硬件 RNE；若实现成先 `.round()` 再 cast，
   在 fp8 上差别极小（0.0265 vs 0.0268），不算错。

## 5. 复跑

```
CUDA_VISIBLE_DEVICES=4 python3 -m harness.baseline --task quant-per-token-fp8-l1 \
    --warmup 30 --iters 200
```
