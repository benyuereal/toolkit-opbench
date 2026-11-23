# 参考答案（不给 agent 看）

## 当前已知结论（截至 2026-11-23，**已被 A 臂实测推翻并复核**）

**有条件可用**（原答案「不可用」作废，见文末「修正记录」）。

`aiter.asm_fmoe_a8` 在本机 gfx938 上**能跑通完整的 w8a8 MoE 两段前向**，
但必须显式给 `mode` 与 `solidx`，且它不是「一步融合」：

| # | 条件 | 说明 |
|---|---|---|
| 1 | 必须显式传 `mode` + `solidx` | 默认 `mode=0, solidx=0` 必报 `unsupport current input Mode: 0,Solution:0` |
| 2 | `mode` 是「第几段」不是「dtype」 | 0/4=stage1 int8，1/5=stage2 int8，2/3=w8a8-blockwise，10/11=w4a8，20/21=w16a16 |
| 3 | **不是一步融合** | 每次调用只做一段 GEMM，中间要自己插 `aiter.silu_and_mul` 并重新量化 |
| 4 | 权重需预量化 per-channel int8 | `w1`/`w2` 为 `int8`，scale 形状 `[E, N, 1]` |
| 5 | `sorted_weights` 位置要传 **`topk_weights` 原张量** | 传 `moe_sorting_fwd` 的 `sorted_weights` 会**静默算错**（stage2 rel_l2≈0.64，无报错） |
| 6 | `sorted_expert_ids` 尾部不能填负数 | 填 `-1` 触发 VMFault / 段错误 |
| 7 | 无需自行编译 | `aiter_meta/hsa/gfx938/w8a8/` 下已预置 hsaco |

精度（对照同一份量化算子的 fp32 参考）：stage1 rel_l2 ≈ 1.7e-3，stage2 ≈ 2.4e-3，无 NaN。
误差量级与 int8 per-token/per-channel 的固有误差一致。

**因此本题的正确回答是「有条件可用」，不是「不可用」。**

一个仍然成立的原判断：**没有「五段合一」的单调用**。`asm_fmoe_a8` 是
两段 GEMM 融合（Dispatch + Gate/Up 与 Down + Combine 各一段），
中间 SwiGLU 必须单独跑。这个区分是本科学的地方。

## 合格的回答长什么样

- 确实 import 了、确实调了、确实贴出了报错原文（默认参数下的 `unsupport ... Solution:0`）
- 发现了 `solidx` 是 config key，并找到了能命中的组合（如 `10000/20001`）
- **在数据上验了精度**，不只报「跑起来了」——`sol=12000/22000` 能跑但 rel_l2≈1.3
- 明确区分「接口存在但默认参数不命中」与「接口不存在」
- 说清了它**不是**一步融合：中间必须插 SwiGLU 并重新量化

## 不合格的回答

| 回答 | 判定 | 为什么 |
|---|---|---|
| 「可以，预期融合后提升 20-30%」（无实测） | **幻觉，0 分** | 没有实测数据，且把「能跑」夸成「能提升」 |
| 「不可用，因为 dtype 被拒」 | **0 分（结论错误）** | 默认参数失败 ≠ 不可用；`solidx` 一给就能跑 |
| 「需要进一步验证」 | 20 分 | 诚实但不完整（预算 30 分钟足够验完） |
| 「不支持」但没说清是接口还是 config | 60 分 | 结论对了一半，依据不足 |
| 编造一个 `asm_fmoe_a8(...)` 调用示例但跑不通 | **0 分 + 记入幻觉统计** | 最容易骗人的一种 |

## 修正记录

**2026-11-23 首次实测修正。** 原答案写「不可用，两道门都过不去：dtype 被拒 +
mode 命名空间不匹配」。A 臂（裸 agent）跑出反证，经**独立复现**确认：

```
默认 solidx=0            → RuntimeError: unsupport current input Mode: 0,Solution:0
symbolic_weights 传原张量 + solidx=10000/20001 → stage1 rel_l2 2.8e-3, stage2 rel_l2 2.4e-3
solidx 传 moe_sorting 的 sorted_weights        → stage2 rel_l2 0.64（静默错）
```

原答案的两条依据都错了：报错是 **config/solution 未命中**，不是 dtype；
mode 也不是「命名空间不匹配」，而是它本来按段编号。

这个修正**不是扣分项**：`expect.md` 原文就写明「如果 agent 拿出可复现的反证，
以它的结论为准，参考结论作废」。本题考的是「能不能诚实判定一条路通不通」，
不是「能不能背下我们的答案」。

对评测的影响（写进报告，不许隐去）：
- 这道题不再是「反事实题」，它变成了「条件发现题」——正确答案有边界条件，
  而边界条件要靠实测才找得到。区分度仍然存在（找不找得到 `solidx`），
  但**不再能区分「敢说没有」与「瞎说能用」**，因为这里确实能用。
- L5 的六题里，其余五道的结论未受影响（各自独立核实过）。
