# 参考答案（不给 agent 看）

## 算子与签名

`causal_conv1d.causal_conv1d_fn(x, weight, bias=None, seq_idx=None,
initial_states=None, return_final_states=False, final_states_out=None,
activation=None)`，版本 **1.5.4+dtk2604.torch2110**（registry 记录 installed: true）。

语义（本机实测与上游 `causal_conv1d_ref` 一致）：

```
out[b, d, t] = sum_{k=0}^{W-1} weight[d, k] * x[b, d, t - (W-1) + k]
```

只在 `t` 及之前取值（causal），序列开头补零；输出与 `x` 同形状 `(B, D, S)`。
本题固定 `B=8, D=2048, W=4`，`S` 取 1024 / 293 / 131。

## 就地改写：实测结论是 **不改写**

这一点必须实测，不能照抄说明。实测（本机 gfx938，bf16，B=8/D=2048/S=1024）：

- 返回值是一个**新 Tensor**，`out.data_ptr() != x.data_ptr()`（`8×2048×1024` 上
  两者分别 140257253130240 / 140257448165376）
- 调用后 `x` 与 `weight` 的内容与原值**逐位相等**（`torch.equal` 为 True）
- 连续调用 6 次后 `x` 仍然不变
- 加 `bias`、`initial_states`、`activation="silu"` 的变体同样不改写 `x`
- 源码佐证：`causal_conv1d_interface.py` 的 `CausalConv1dFn.forward` 内
  `causal_conv1d_fwd_function` 里 `out = torch.empty_like(x)` 后由 C++ 写入 out；
  `mutates_args={"out", "final_states_out"}` 只声明输出与 final_states_out 可变，
  **不含 x**。

所以 `definition.dcu.inplace_inputs` **不声明**（根本没有原地改写）。若某版本改成
原地语义，以实测为准，definition 与流量口径跟着改。

## 流量口径：读 1 份 x + 读 weight + 写 1 份 out = `4*B*D*S + 2*D*W` 字节

逐项（每元素 2 字节，bf16）：

| 项 | 字节 | 说明 |
|---|---|---|
| 读 x | `2*B*D*S` | 每个输入元素被 4 个输出窗口复用（W=4 的重叠窗口），但**从 DRAM 只取一次**，之后在片上复用 |
| 读 weight | `2*D*W` | `D×W`，很小（2048×4×2 = 16 KB），但它是算子**必须读**的一份参数，不能忽略 |
| 写 out | `2*B*D*S` | 输出整块落盘 |

合计 `4*B*D*S + 2*D*W`。以 S=1024 为例：`4*8*2048*1024 + 2*2048*4 = 67,108,864 + 16,384`
≈ **67.1 MB**。

### 两个必须写清的判断（这是本题的核心）

1. **不算「读 4 份 x」。** W=4 的因果卷积里每个输入被 4 个输出窗口逻辑上覆盖，
   但相邻输出复用同一批输入——只要实现把输入块留在寄存器/共享内存里，每个元素
   从 DRAM 只读一次。把逻辑访问次数当 DRAM 流量，会把读量算成 4 倍，达成率被
   压到 ~25%，得出"还有 4 倍空间"的错误结论。
2. **causal 掩码「一半窗口为零」不减流量。** 掩码削掉的是**计算量**（乘法/加法次数），
   不是访存量：每个输入元素该读的还得读，输出每个位置都要算、都要写。流量仍按
   全量 `[B, D, S]` 计。把掩码算成"流量减半"会虚高近一倍。

> 注意 `weight` 那一项 `2*D*W` 相对 `4*B*D*S` 极小（16 KB vs 数 MB），算不算都几乎
> 不影响达成率，但**口径上必须计入**——它是算子的真实读流量。本题保留它，
> 也是提醒「小算子不要把固定参数读量和主流量混为一谈」。

## 分母口径：`peak_achievable:copy` = 1350 GB/s

- 这个算子几乎**纯访存**：W=4 的 depthwise 卷积每元素只有 4 次乘加，
  算术强度 ≈ 8 flop / 4 byte = 2 flop/byte，远低于本机 bf16 的拐点
  （279 TF / 1350 GB/s ≈ 207 flop/byte）。它离算力上限极远，**带宽**才是瓶颈。
- 分母用 `peak_achievable:copy`（1350 GB/s，本机 copy R+W 实测峰值，可复跑），
  而不是 `spec_bandwidth`（1100，型号标称，不可复跑）。用 1100 当分母会把达成率
  算虚高约 23%（1350/1100 ≈ 1.23）。
- **为什么是 copy 而不是 read**：本算子的流量是「读一块 + 写一块」，结构上就是
  一次流式 copy，与 `peak_achievable:copy`（同时含读与写）同口径。`peak_achievable:read`
  = 1724 GB/s 是**纯读**上限；本体要写出整块 out，用纯读上限当分母会高估上限、
  压低达成率，且口径与实际读写比不符。故用 copy。

`dcu.work_unit = "byte"`，单位与 `peak_achievable:copy` 的 GB/s 匹配（baseline
会校验，不匹配直接报错）。

## 形状（三个，含不整齐的）

- public：`S=1024`、`S=293`
- holdout：`S=131`

`S=293 / 131` 是刻意不整齐（非 8 的倍数、非对齐），破「刚好对齐」的写法，
也逼近线性注意力常见的短序列 decode/chunk 边界。
另：短序列下**单次 kernel launch 固定代价**占比会显著上升，是本题除流量口径外的
第二个真陷阱——但题面只问带宽达成率，agent 若发现短序列达成率明显低于长序列并
归因到 launch，是加分点。

## 参考数值（本次实测，仅供对照，不许抄进 baseline，最终以 baselines/*.json 为准）

见 `baselines/lin-causal-conv1d-l1.json` 的 `t_b_ms` / `t_sol_ms` / `shapes`。
基线实现就是 `causal_conv1d.causal_conv1d_fn` 本身（厂商实现），不是 torch 手写。

## 合格的回答长什么样

- 落到 `causal_conv1d.causal_conv1d_fn`，写明版本与完整签名
- 明确流量是 **1读 x + 读 weight + 1写 out**，并说明「W 重叠不复读」「掩码不减流量」
- 分母用 `peak_achievable:copy` 1350，并说明为什么是带宽口径、为什么用 copy 而非 read
- 三个 S 都有数字（含不整齐的 293/131）
- 用 GPU event 计时，处理了 L2 flush
- 能指出短序列下 launch 占比上升（若观察到）

## 不合格的回答

| 回答 | 判定 | 为什么 |
|---|---|---|
| 按「读 4 份 x」算流量，达成率 ~25% 说"还有 4 倍空间" | **0 分** | 把逻辑访问当 DRAM 流量，分母/分子口径双错 |
| 按 causal 掩码把流量减半，达成率虚高近一倍 | **0 分** | 掩码削计算量不削访存 |
| 用 `peak_achievable:read`(1724) 当分母 | 60 分 | 本算子要写整块 out，纯读上限口径不符 |
| 用 `spec_bandwidth`(1100) 当分母，报"已达标称带宽 90%+" | 60 分 | 分母用了不可复跑的标称值 |
| 用 torch `F.conv1d` 手写当被测实现 | 60 分 | 测的是参考实现，不是"当前实现" |
| 只测一个 S，挑好看的报 | 60 分 | 单点不支撑结论 |
| 用 `time.time()` 量短序列，报 0.0 ms | **0 分** | 计时方法错误 |

## 一个必须允许的例外

如果 agent 实测发现本机某版本 `causal_conv1d_fn` **会**原地改写输入（例如某次
DTK 升级改了语义），以它为准：definition 补 `inplace_inputs: ["x"]`，流量口径
跟着实测走，不能跟着本文档走。同理若实测流量项与本文档不符（例如某实现在
短序列上有一份额外 workspace 写），以可复跑的实测为准并在报告里说明依据。
