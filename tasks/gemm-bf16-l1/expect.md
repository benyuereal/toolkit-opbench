# 参考答案（不给 agent 看）

## 当前已知结论（截至 2026-11-23）

**bf16 大 GEMM 在本机已基本打满，没有多少空间。**

- `peak_achievable:bf16` = 279 TFLOPS（本机 hipBLASLt 大 GEMM 10 轮独立采样峰值中位数，
  status=measured，at_sclk_mhz=1350，reproduce 见 constants.yaml）
- 实测同样走 hipBLASLt 的实现，4096³ 量级上达成率约 **95%+**
- 结论：**已到顶，该去优化别的地方**（例如融合、减少 launch、访存）

依据要点：
1. 分母是 `peak_achievable`（279），**不是** `peak_nominal`（280）。
   两者在当前差值小，但语义不同：nominal 不可复跑，achievable 可复跑。
2. 279 是在 **sclk=1350 MHz（档位 8）** 上量的。降到 600 MHz 会差一倍多。
   所以每次测量必须记 sclk。
3. 不能拿 `impl_measured:fp8`（49.8）当分母——那是 aiter opus 族的值，
   与 bf16 无关，且是"某条实现"不是"硬件"。

## 合格的回答长什么样

- 落到具体实现（`torch.mm` / hipBLASLt 或其它），写明版本
- 至少两个 M 档位都有数字，不是只报最好的那个
- 明确说分母取自 `constants.yaml` 的 `peak_achievable:bf16`，并说明为什么
- 结论是「已到顶 / 空间很小」，并给出下一步该往哪走的判断
- 记了 sclk

## 不合格的回答

| 回答 | 判定 | 为什么 |
|---|---|---|
| 「还能快 10 倍，建议换 fp8」 | **0 分** | 无依据，且远超当前实测空间 |
| 「达成率 100%」但分母写的是标称 280 | 60 分 | 方向对，分母口径混了 nominal/achievable |
| 「已到顶」但只测了一个形状 | 60 分 | 单点不能支撑"到顶" |
| 用 `time.time()` 量，报出 0.0 ms 或极大值 | **0 分** | 计时方法错误 |
| 没处理 L2 缓存 | 60 分 | 测出来的可能虚高，但结论方向未必错 |
