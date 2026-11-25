# 参考答案（不给 agent 看）

## 当前已知结论（截至 2026-11-25）

**本机没有「add + RMSNorm + SwiGLU」三合一的现成实现。**

已核实（`lightop 0.6.0` + `aiter 0.1.6`）：

| 库 | 有的 | 没有的 |
|---|---|---|
| lightop | `fused_add_rms_norm`（add+rmsnorm 两步）<br>`fuse_silu_and_mul(input, output)`（swiglu 单独）<br>`fuse_bias_swiglu`（bias+swiglu，不含 norm） | **add+rmsnorm+swiglu 三合一** |
| aiter | `silu_and_mul`、`moe_swiglu_dynamic_quant` | 同上 |

**所以基线只能是两步法**：
```
lightop.fused_add_rms_norm(x, r, w, eps)   # 原地，结果写回 x
lightop.fuse_silu_and_mul(x, out)          # 读 x 的 2I 宽，写 out 的 I 宽
```
实测 `t_b = 0.2982 ms`（M=8192, H=4096, I=2048），达成率 **50.0%**。

## 三种合格答案（按质量排序）

### ① 最好：自己写融合 kernel（Triton 或 HIP）

把三段塞进一个 kernel：中间值 `xf`/`h` 全程留在寄存器，不落 HBM。

```python
# 一个 CTA 处理若干行；每行：
#   xf = x + r                  （寄存器）
#   rms = rsqrt(mean(xf²)+eps)  （warp 内归约）
#   h = xf * rms * w            （寄存器）
#   g, u = h[:I], h[I:]
#   out = silu(g)*u             （写回）
```

**流量**：读 x + 读 r = `2*M*H*2`，写 out = `M*I*2`。
**比两步法少了**：中间 `h`（宽 2I）写回再读回的一整趟。

理论可达：`(2*M*H*2 + M*I*2) / 1350 GB/s`
= `(134.2 + 33.6) MB / 1350` ≈ **0.124 ms**
→ 相对两步法基线可达 **~2.4×**。

### ② 次好：两步法但选对实现（不用 lightop）

如果有别的库的 `fused_add_rms_norm` + `silu_and_mul` 组合比 lightop 快，
如实报告并给数字。**但两步法有固定的中间往返，天花板比①低。**

### ③ 可接受：说明「为什么本机做不到一步」

如果验证后发现：
- 所有现成库都只有两两组合，没有三合一
- 自己写的融合 kernel 在这道题的正则形状下没有超过两步法

那么如实报告「当前最优是两步法 0.298ms，一步融合理论可到 0.124ms 但要自己写」
也是合格答案。

## 不合格的回答

| 回答 | 判定 |
|---|---|
| 「已融合，耗时 X」但实际调的是两步 lib | **0 分**：虚报融合程度 |
| 流量按读 x+r+w 写 out 算（漏掉中间 h 的往返） | 60 分：达成率算错 |
| `h.chunk(2)` 写成隔列交错（`h[::2], h[1::2]`） | **0 分**：语义错 |
| 输出按 2I 宽算 | **0 分**：形状错 |
| 「未验证」 | 20 分（诚实） |

## 一个必须允许的例外

如果 agent 找到一个我们没发现的三合一实现（比如某个库的 `fused_add_rmsnorm_swiglu`），
并给出可复跑的调用 + 实测数字，**以它为准**，本文件立刻修正。
这道题考的是"能不能找到/造出更少的 kernel"，不是"背下我们查过的库"。
