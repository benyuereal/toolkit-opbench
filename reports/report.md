# toolkit-opbench 报告（三臂对照）

## 结论一句话

> Δ_eff(C−A) < 20pt ⇒ 该砍文档，不是加文档；B 与 C 差 < 10pt ⇒ 判据层无价值，只留库清单；Δ_halluc(C−A) > 0 ⇒ 立刻停，skill 在教模型编答案

## 三臂对比

| 题 | 臂 | 状态 | 有效结论 | judgment | 幻觉率 | 泄漏 | 墙钟 s | 产物 |
|---|---|---|---|---|---|---|---|---|
| moe-gating-l1 | A | ok | ✓ | — | 0.03 | 0 | 1140.0 | 3 |
| moe-gating-l1 | B | ok | ✓ | — | 0.06 | 0 | 286.5 | 4 |
| moe-gating-l1 | C | ok | ✓ | — | 0.05 | 0 | 459.3 | 5 |
| norm-fused-add-rmsnorm-l1 | A | ok | ✓ | — | 0.00 | 0 | 1313.5 | 12 |
| norm-fused-add-rmsnorm-l1 | B | ok | ✓ | — | 0.00 | 0 | 705.1 | 10 |
| norm-fused-add-rmsnorm-l1 | C | ok | ✓ | — | 0.00 | 0 | 1247.6 | 5 |

## Δ

| 指标 | A | B | C |
|---|---|---|---|
| 题数 | 2 | 2 | 2 |
| 有效结论率 % | 100.0 | 100.0 | 100.0 |
| judgment 均分 | — | — | — |
| 幻觉率 | 0.0 | 0.0 | 0.0 |

- Δ_eff C−A (pt): +0.00
- Δ_eff B−A (pt): +0.00
- Δ_eff C−B (pt): +0.00
- Δ_halluc C−A: +0.01
- Δ_halluc B−A: +0.01

## 逐题判定理由

- **moe-gating-l1 / A** — 有效结论：有命令 + 数字/原始输出
- **moe-gating-l1 / B** — 有效结论：有命令 + 数字/原始输出
- **moe-gating-l1 / C** — 有效结论：有命令 + 数字/原始输出
- **norm-fused-add-rmsnorm-l1 / A** — 有效结论：有命令 + 数字/原始输出
- **norm-fused-add-rmsnorm-l1 / B** — 有效结论：有命令 + 数字/原始输出
- **norm-fused-add-rmsnorm-l1 / C** — 有效结论：有命令 + 数字/原始输出

## 泄漏与污染

- 未命中泄漏指纹。
- 时钟：见 runs/trace.jsonl 的 sclk_mhz（L5 题不碰 GPU，为空）
