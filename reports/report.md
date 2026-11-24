# toolkit-opbench 报告 — L1 三臂对照（moe-gating / rmsnorm）

> 生成日期：2026-11-24
> 方法：**性能数据全部来自 `harness.measure` 统一口径实测**，
> 不是从三臂各自的产物里拼的（原因见文末「为什么不用自报数」）。

---

## 一、结论一句话

> **两道 L1 题都没测出 skill 的价值。**
> moe-gating 上 B/C 并列（差 0.3%），A 落后；rmsnorm 上三臂完全相同。
> 按预注册的胜负线（`B 与 C 差 < 10pt ⇒ 判据层无价值`），这两题支持「只留库清单」。

---

## 二、性能（bench 主指标，权重 80%）

正确性是**硬门槛**：三臂在两题上都通过（见第四节），所以都进入本表。

| 题 | 形状 | A | B | C | 最快 |
|---|---|---|---|---|---|
| moe-gating-l1 | M=16384, E=64, k=8 | 134.40 µs (43.7%) | 58.88 µs (99.7%) | **58.72 µs (100%)** | C |
| norm-fused-add-rmsnorm-l1 | M=8192, H=4096 | 0.1960 ms (100%) | 0.1960 ms (100%) | 0.1960 ms (100%) | 三臂相同 |

### roofline 判定

| 题 | 带宽地板 | launch 地板 | 实测 | 支配项 | 结论 |
|---|---|---|---|---|---|
| moe-gating | 2.53 µs | **43 µs** | 58.7 µs | **launch** | 已在 1.4× 地板内，**优化 kernel 内部无意义** |
| rmsnorm | 0.1988 ms | 0.043 ms | 0.1960 ms | bandwidth | 1370 GB/s，**已打满**（copy 口径 1350 的 101%） |

**moe-gating 的物理事实**：4.98 MB 的活，带宽上只需 2.5 µs，但单次 launch 就要 43 µs。
三臂实测 58.7–134 µs，**全部被 launch 支配**。这道题正确的答案本该是
「别优化这个 kernel，去融合或减 launch」，而不是比谁快几微秒。

---

## 三、实现落点（人工核实，非自动）

| 题 | A | B | C |
|---|---|---|---|
| moe-gating | **自写 Triton**（三个变体，最好 134µs） | `sgl_kernel.topk_softmax` | `aiter.moe_c_topk_softmax` |
| rmsnorm | `lightop.fused_add_rms_norm` | `aiter.fused_add_rmsnorm_cu` | `lightop.fused_add_rms_norm` |

**A 臂在 gating 上判错了一个事实**：它说「库里没有现成的融合 top-k+softmax 单算子」，
只查了 `torch.ops.aten`。实际本机有：
- `aiter.moe_c_topk_softmax`（C 臂用的）
- `sgl_kernel.topk_softmax`（B 臂用的）
- `lightop` 里 11 个 topk/gate 相关入口

这条错误**直接导致了 A 慢 2.3 倍**——它自己造轮子，而现成实现贴着 launch 地板。

---

## 四、正确性（硬门槛，全部通过）

我重新跑了三臂的**实际实现**（不看它们的自述）与题目 reference 对比：

| 臂 | 实现 | max_abs | 结论 |
|---|---|---|---|
| A | 自写 Triton（3 变体） | 1.8e-07 | ✅ 通过 |
| B | `sgl_kernel.topk_softmax` | 1.8e-07 | ✅ 通过 |
| C | `aiter.moe_c_topk_softmax` | 1.8e-07 | ✅ 通过 |

rmsnorm 三臂都落到 `lightop`，语义正确（2读2写，`residual = x + residual`）。

---

## 五、效率（迭代 / token / 成本）

| 题 | 臂 | num_turns | 输出 token | 成本 | 墙钟 |
|---|---|---|---|---|---|
| moe-gating | A | 37 | 31,992 | $1.50 | 1140 s |
| | B | **17** | **8,547** | **$0.53** | **287 s** |
| | C | 42 | 14,942 | $1.14 | 459 s |
| rmsnorm | A | 64 | 56,295 | $2.73 | 1314 s |
| | B | 37 | 24,946 | $1.30 | 705 s |
| | C | 36 | **63,823** | $2.33 | 1248 s |

**B 最省**（gating 上迭代数是 C 的 40%）。C 在 gating 上迭代最多（42 次）——
它读了 skill 后仍在反复验证。**skill 让它更谨慎，不是更快。**

---

## 六、Δ 与胜负线

```
Δ_eff   C−A = +0 pt    B−A = +0 pt      （有效结论率三臂 100%）
Δ_halluc C−A = +0.01   B−A = +0.01      （三臂幻觉率均 ≈0.03–0.06，差异在噪声内）
性能     C−A = +56 pt  B−A = +56 pt（gating）
        C−B = +0.3 pt
```

按预注册规则（**跑前定死，跑后不改**）：

| 条件 | 是否触发 | 含义 |
|---|---|---|
| `B 与 C 差 < 10pt` | ✅ **触发** | 判据层无价值，只留库清单 |
| `Δ_eff(C−A) < 20pt` | ✅ 触发 | 有效结论率无提升 |
| `Δ_halluc > 0` | ❌ 未触发 | 没有证据显示 skill 在教模型编答案 |

---

## 七、这两个结论的可信度

**样本 n=2，每臂单次运行。这是方向，不是判决。**

更重要的限制：**这两道题本身不适合测 skill。**

- **moe-gating**：被 launch 支配（43µs 地板）。三臂都在 1.4× 地板内，
  性能差异的空间本身就很小——再怎么选实现也就差几微秒。
- **rmsnorm**：三臂都找到了 `lightop.fused_add_rms_norm` 这个正确实现，
  1370 GB/s 已经打满。**没有给 C 臂留任何发挥空间。**

真正能区分 skill 的题，答案必须**无法靠 `dir()` 或读一个 API 文档得到**。
已建好 `suites/skill-probe.yaml` 用于此目的（见第八节）。

---

## 八、下一步

| 动作 | 为什么 |
|---|---|
| 跑 `suites/skill-probe.yaml` 三臂 | 三道题的答案都要靠「本机踩过的坑」或判据 |
| 每题 ≥3 次重复 | 消除单次方差，才能从"方向"变"结论" |
| 权重落地到自动评分 | 现在性能 80% 是手工按统一实测算的，要接进 collect |

---

## 附录：为什么不用三臂自报的数

我最初想直接从各臂的 `results/*.json` 里抽耗时算排名，实测发现**不可比**：

| 臂 | 字段名 | 单位 | 备注 |
|---|---|---|---|
| A | 无 JSON（只有 `.txt` 日志） | — | — |
| B | `median_s` | **秒** | 有的带 flush 有的不带 |
| C | `rows[i].fused.median_ms` | 毫秒 | 逐次 event |

同一个 `lightop.fused_add_rms_norm`，B 报 606 µs、C 报 203 µs，
**我统一实测 196 µs**。拼出来的排名一半是口径差异，不是实现差异。

还踩过两个坑：
- B 的 `results` 里有 `empty_loop_ms = 0.00019`（空循环基准），
  被当成 gating 耗时 → 显示"快 228 倍"。
- 不同形状混比（M=1024 的 38 µs vs M=8192 的 203 µs）→ 假排名。

所以本报告的性能数据**一律用 `harness.measure` 重测**：
逐次 CUDA event、固定 warmup/iters、每轮 flush L2、
原地改写轮换 buffer、记录 sclk。各臂自报的数只用于判「有没有如实报告」。
