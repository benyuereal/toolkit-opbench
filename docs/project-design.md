# toolkit-opbench · 项目设计（待审）

> 目标：**测增益，不跑分。** 同等预算下，带 skill 的 agent 比裸 Claude Code 好多少。
> 允许结论是「没更好」——那就该砍文档，而不是继续加。

---

## 一、目录结构

```
toolkit-opbench/
├── README.md                    项目是什么、怎么跑、三条铁律
├── protocol.md                  操作协议：隔离 / GPU 互斥 / 评分 / 防作弊
├── docs/
│   ├── design.md                为什么这么测（三臂、为什么不打榜）
│   ├── taxonomy.md              算子分类 + 五层定义 + 覆盖矩阵
│   ├── scoring.md               评分口径（重跑、幻觉机检、泄漏指纹）
│   └── report-template.md       报告格式
├── tasks/                       题库，一题一目录
│   ├── _TEMPLATE/
│   │   ├── task.md              题面（三臂逐字相同）
│   │   ├── expect.md            参考答案（不给 agent 看）
│   │   └── check.py             机检脚本（可选的自动判定）
│   ├── gemm-bf16-l1/
│   ├── ...
├── suites/                      每次跑的子集（题库大，按套跑）
│   ├── l5-only.yaml             6 道反事实题，最便宜，先跑这个
│   ├── smoke.yaml               6 题快速回归
│   └── full.yaml                全量，分批跑
├── arms/
│   ├── run.sh                   单臂单题执行器
│   └── contexts/{a,b,c}.md      三臂各自的上下文注入说明
├── scorer/
│   ├── score.py                 主评分（重跑 + 汇总）
│   ├── hallucination.py         幻觉机检
│   └── leak_detect.py           A 臂泄漏指纹检测
├── runs/                        运行产物（gitignored）
└── reports/                     每次评测的报告
```

**设计要点**：题库与运行子集分离。36 道题不可能每次都跑，所以用 `suites/` 定义"这次跑什么"。

---

## 二、五个层级（task 的 tier 字段）

| tier | 考什么 | 判据形态 | 预算 |
|---|---|---|---|
| **L1** 已覆盖 | atlas 里有卡且已有实测 → 知识层能不能直接用上 | 能否落到 `库.函数` 并跑出可复跑数字 | 30 min |
| **L2** 未覆盖 | atlas 里没有或只有 draft → 迁移能力 | 允许"查不到"，但要说清查了什么 | 30 min |
| **L3** 边界决策 | 融合/拆分、吸收/非吸收、重叠/串行 | **给拐点或选择规则**，不是给结论 | 45 min |
| **L4** 端到端 | 真实仓库改一处，不能回归 | 端到端指标 + 无回归 | 120 min |
| **L5** 反事实 | 正确答案是「不通 / 别做」 | 答「不通 + 机制」满分；答「能优化 X%」= 0 | **10 min** |

**L5 是这个项目的核心创新**：外部榜假设题目一定有解，而我们的真实工作一大半在判断"这条路通不通"。

---

## 三、算子分类（9 个家族）

| 家族 | 覆盖什么 | 为什么单列 |
|---|---|---|
| `gemm` | GEMM 各精度档、量化前置、epilogue 融合 | 所有算力的底座，MoE 里占 89–96% |
| `moe` | 路由、align/permute、grouped GEMM、融合边界、EP | 唯一"多段"结构，pipeline 能力的主要载体 |
| `attn-dense` | MHA/GQA/FlashAttn/paged decode | 推理热路径 |
| `attn-mla` | MLA 吸收/非吸收、稀疏 MLA、mqa-logits | DeepSeek/GLM 系专用 |
| `norm-act` | RMSNorm、fused-add、SwiGLU/GELU | 带宽受限类，最适合做校准件 |
| `kv` | paged write、layout、KV 与计算重叠 | 长上下文的关键 |
| `linear-attn` | causal-conv1d、MHC 三件套、hadamard | 本地有已优化资产，测"能否复用" |
| `head-sample` | LM head、top-k/top-p 采样 | 末段，常被忽略但有实测价值 |
| `pipeline` | 段占比、重叠判别、通信归层 | 跨算子决策，`layer: engine` 的落点 |

---

## 四、覆盖矩阵（家族 × 层级）

| 家族 | L1 | L2 | L3 | L4 | L5 | 合计 |
|---|---|---|---|---|---|---|
| gemm | 2 | 2 | 1 | — | 1 | 6 |
| moe | 3 | — | 1 | 1 | 2 | 7 |
| attn-dense | 2 | 1 | — | 1 | 1 | 5 |
| attn-mla | 2 | 1 | 1 | 1 | — | 5 |
| norm-act | 2 | 2 | — | — | — | 4 |
| kv | 1 | 1 | 1 | 1 | — | 4 |
| linear-attn | 1 | 1 | — | — | 1 | 3 |
| head-sample | 1 | 1 | — | — | — | 2 |
| pipeline | — | — | 2 | — | 1 | 3 |
| **合计** | **14** | **9** | **6** | **4** | **6** | **39** |

**覆盖判断**：9 个家族各有题目，5 个层级都有题；L5 六题覆盖 gemm/moe/attn/linear-attn/pipeline 五个家族——这五个恰好是我们历史上"编过结论"的地方。

---

## 五、题库（39 题清单）

### gemm（6）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `gemm-bf16-l1` | L1 | bf16 大形状达成率，**分母必须用实测峰值** | ✓ | 给 ratio + 分母来源 + 是否到顶 |
| `gemm-int8-l2` | L2 | int8 分组 GEMM 的路子 | ✓ | 指出 deepgemm i8 入口或其缺失 |
| `gemm-batched-l2` | L2 | batched vs grouped 的选择 | ✓ | 选择规则 + 依据 |
| `gemm-quant-per-token-l1` | L1 | per-token 量化（FP8 前置） | ✓ | 函数 + 与 gemm-fp8 的依赖关系 |
| `gemm-swiglu-epilogue-l3` | L3 | epilogue 融合的 M 拐点 | ✓ | **给拐点**，不给"融合更快" |
| `gemm-fp8-l5` | **L5** | fp8 是否原生可用 | ✓ | **「本机走 dequant 回退，fp8 慢于 bf16 即证」** |

### moe（7）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `moe-gating-l1` | L1 | top-k gating 的实现选型 | ✓ | `lightop.fuse_topk_transform` 一族 |
| `moe-align-permute-l1` | L1 | align + permute | ✓ | 函数 + 形状约束 |
| `moe-grouped-gemm-w8a8-l1` | L1 | w8a8 grouped GEMM | ✓ | 函数 + **N 必须 256 倍数**等硬约束 |
| `moe-fusion-boundary-l3` | L3 | 五段融合的 M* | ✓ | **M/exp < 32 用部分融合，> 64 用全拆分** |
| `moe-end2end-combine-l4` | L4 | EP combine 端到端 | ✓ | 端到端指标 + 无回归 |
| `moe-asm-fmoe-l5` | **L5** | `aiter.asm_fmoe_a8` 可用性 | ✗ | **「全 dtype 被拒 + mode 命名空间不匹配」** |
| `moe-shared-expert-l5` | **L5** | 共享专家能否双流并行 | ✓ | **「同引擎，重叠 0%，并发不成立」** |

### attn-dense（5）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `attn-paged-decode-l1` | L1 | paged decode 达成率 | ✓ | **识别 ratio 极低（≈0.03）意味着要么大空间要么口径错** |
| `attn-flash-fwd-l1` | L1 | 等 FLOPs 下两实现对比 | ✓ | 指出 flash-attn 在等 FLOPs 下是 `avoid` |
| `attn-prefill-split-l2` | L2 | split-k / split-prefill 选择 | ✓ | 选择规则 + 阈值 |
| `attn-end2end-mla-patch-l4` | L4 | 真实仓库改一处注意力 | ✓ | 端到端 + 无回归 |
| `attn-gqa-repeat-kv-l5` | **L5** | 要不要为 repeat_kv 建实现 | ✗ | **「别做，纯 broadcast，写进 not_for」** |

### attn-mla（5）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `mla-sparse-l1` | L1 | 稀疏 MLA 的地板类型 | ✓ | **指令数地板，不是带宽地板** |
| `mla-mqa-logits-l1` | L1 | mqa logits 达成率 | ✓ | ratio + 对照口径 |
| `mla-kv-layout-l2` | L2 | MLA KV cache 布局 | ✓ | 布局选择 + 代价 |
| `mla-absorb-vs-full-l3` | L3 | 吸收 vs 非吸收选择 | ✓ | **等 FLOPs 下非吸收快 4.54×** |
| `mla-end2end-attn-l4` | L4 | 端到端接入 | ✓ | 端到端 + 无回归 |

### norm-act（4）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `norm-fused-add-rmsnorm-l1` | L1 | 样板卡（九件套齐全） | ✓ | 三闸全过 + 实测 |
| `norm-rmsnorm-variant-l2` | L2 | 未见变体（如 RMSNorm+Dropout） | ✓ | 迁移：能否类推 + 验证 |
| `act-swiglu-fused-l1` | L1 | 融合 vs 拆分 | ✓ | **2.25×，达成率 46%** |
| `act-gelu-geglu-l2` | L2 | GELU 变体选型 | ✓ | 函数 + 场景 |

### kv（4）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `kv-paged-write-l1` | L1 | paged write | ✓ | 函数 + 约束 |
| `kv-layout-l2` | L2 | layout 选择 | ✓ | 规则 |
| `kv-copy-overlap-l3` | L3 | KV 备份与计算能否重叠 | ✓ | **只有真 DMA 能重叠（100%），D2D copy 不行（11.9%）** |
| `kv-end2end-hicache-l4` | L4 | HiCache 场景端到端 | ✓ | 端到端 + 无回归 |

### linear-attn（3）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `lin-causal-conv1d-l1` | L1 | causal-conv1d | ✓ | 函数 + 与 MHC 的关系 |
| `lin-mhc-fused-l2` | L2 | MHC 融合（本地有已优化资产） | ✓ | 能否复用已有 patch |
| `lin-hadamard-l5` | **L5** | hadamard 有没有独立 symbol | ✗ | **「没有，内联在 mhc_pre_mix_kernel 里」** |

### head-sample（2）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `head-lm-sampling-l1` | L1 | lm_head + sampling 合并 | ✓ | 为什么合成一张卡 |
| `head-topk-topp-l2` | L2 | 采样算子选型 | ✓ | 函数 + 场景 |

### pipeline（3）

| id | tier | 考什么 | 需 GPU | 参考答案形态 |
|---|---|---|---|---|
| `pipe-segments-l3` | L3 | MoE 五段段占比 | ✓ | **GEMM 占 89–96%，融合上限 4–11%** |
| `pipe-overlap-discrim-l3` | L3 | 重叠判别法 | ✓ | **用 min 做分母，区分真/伪/无重叠** |
| `pipe-allreduce-l5` | **L5** | 通信该不该建算子卡 | ✗ | **「不该，kernel 属 RCCL，能动的是重叠编排」** |

---

## 六、三套运行子集

| suite | 含 | 单臂耗时 | 用途 |
|---|---|---|---|
| **`l5-only`** | 6 道 L5 | ~1 h | **第一次就跑这个**：不用 bench，最便宜，直接暴露"会不会编" |
| `smoke` | 2 L5 + 3 L1 + 1 L3 | ~2.5 h | 每次改 skill 后的回归 |
| `full` | 39 道 | 分批（约 3 人天/臂） | 出正式结论 |

**L5 单独成套**是刻意的：它能在没有任何 GPU 争用的情况下，先回答"这套东西会不会教模型编答案"。

---

## 七、评分口径（摘要，详见 docs/scoring.md）

| 指标 | 判定 | 权重 |
|---|---|---|
| 有效结论率 | 有可复跑数字 + 复跑方法 | 30% |
| 正确性 | 重跑通过 + 未见 shape + 压力数值 | 25% |
| 跨 shape 几何平均加速比 | 对基线 | 15% |
| **校准度** | 宣称达成率与实测差 ≤ 10pt | 15% |
| **幻觉率** | 编造 `import_path`/库名/行号（逐条 import 机检） | 15% |

L5 单独评：答「不通 + 机制」100；「不知道」20；「能优化 X%」0 + 计入幻觉。

---

## 八、防作弊（摘要）

1. 预算固定（wall time + token）
2. 评分器只读产物，**不接受自述**
3. 评测 shape 与题面 shape 差 20%（留出）
4. L2 题在 C 臂的 skill 里**不能提前写卡**（否则泄题）
5. A 臂隔离：`opeval` 用户 + skill 仓库 `chmod 700`（已实测 BLOCKED）
6. A 臂泄漏指纹检测（6 条指纹，命中即作废）

---

## 九、实施顺序

| 阶段 | 做什么 | 产出 |
|---|---|---|
| **P0** | `l5-only` 六题 + 评分器 + 泄漏检测 | 第一次 Δ（A/B/C 各 6 题） |
| P1 | `smoke` 六题 | 可回归的最小集 |
| P2 | 补齐 L1/L2 题（需 bench） | 23 题 |
| P3 | L3 + L4（需真实仓库环境） | 10 题 |

**先做 P0 的理由**：不用 GPU、不用真实仓库、每题 10 分钟，一晚能出第一个结论。

---

## 十、三个需要你拍板的点

1. **仓库私有还是公开？** 我建议 **private**：题面里有内部形状，expect.md 里有我们的结论（含失败案例）。
2. **C 臂能不能直接读 `/workspace/toolkit_skills`？** 建议能（只读），这才是"完整 skill"；同时它也是被测对象，改它要另开 commit。
3. **第一轮预算**：`l5-only` 六题 × 三臂 = 18 次 headless 调用，每次 10 分钟，总计约 **3 小时**，今晚能跑完。是否按这个走？
