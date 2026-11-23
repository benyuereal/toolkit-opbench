# toolkit-opbench

> **算子级性能评测台。** 它不跑分，测的是**增益**：同等预算下，带 skill 的 agent 比裸 Claude Code 好多少。

面向海光 DCU（gfx938）的大模型算子优化场景。被测对象是 [`toolkit-skills`](../toolkit-skills) 那套知识层，本仓库是**测量工具**。

---

## 一、这个仓库回答一个问题

> 我们那套 46 张 atlas 卡 / 13 个家族 / 6 个场景，**提高了产出，还是只让文档变厚？**

**必须允许答案是「没提高」。** 如果 A 臂（裸）与 C 臂（完整 skill）分不出差别，那该砍的是文档，而不是继续加文档。

配套的三个子问题：

| 问题 | 由谁回答 |
|---|---|
| 带 skill 是否更少编造？ | `suites/l5-only.yaml`（6 道反事实题，不用 GPU） |
| 值钱的是「知道在哪」还是「知道怎么判断」？ | B 臂 vs C 臂 |
| 这套东西该不该继续投？ | 三个 Δ 与跑前定死的胜负线 |

---

## 二、架构：三层，职责不能混

```
┌──────────────────────────────────────────────────────────────┐
│  harness/            实验驱动：比得公平                       │
│  run.py  arms/{a,b,c}.py  collect.py  doctor.py              │
│  · 三臂上下文隔离   · 预算双限（wall time + token）           │
│  · A 臂换用户跑（读不到 skill）  · 超时不续跑                  │
└───────────────────────────┬──────────────────────────────────┘
                            │ 调用：怎么量得准
┌───────────────────────────▼──────────────────────────────────┐
│  bench/              测量原语：量得准                         │
│  data/    definition · workload · solution · trace           │
│  hw/      dcu（卡数/L2/时钟）· constants（峰值，禁字面量）      │
│  timing/  event · flush（L2）· stats（median/MAD/重叠度）      │
│  check/   correctness · integrity · leak · hallucination     │
│  score/   ratio · sol_like · judgment · calibrate            │
└───────────────────────────┬──────────────────────────────────┘
                            │ 输入：量什么
┌───────────────────────────▼──────────────────────────────────┐
│  tasks/  suites/  baselines/                                  │
│  题面三件套 + 运行子集 + 基线存档                              │
└──────────────────────────────────────────────────────────────┘
```

**为什么把 `bench` 和 `harness` 分开**：`bench` 管「量得准」，`harness` 管「比得公平」。混在一起，每次改实验设计都会动到测量代码，历史数据立刻不可比。

---

## 三、一次评测的数据流

```
tasks/<id>/{task.md, definition.json, workload.jsonl, reference.py}
        │
        ├── prepare ──→ (task, arm) 队列：A / B / C × 每题
        │
        ├── run ─────→ runs/<arm>/<task>/attempt-N/{transcript.json, artifacts/}
        │                 A 臂：su - opbench（chmod 700 挡住 skill）
        │                 B 臂：+ libraries-registry.yaml
        │                 C 臂：+ skill/（只读软链）
        │
        ├── verify ──→ 产物齐全？题面未改？无劫持？重跑数值对得上？import 真存在？
        │                 任一项失败 → 0 分 / 作废，不进 score
        │
        ├── score ───→ ratio（主指标）· sol_like（只报告）· judgment（L3/L5）
        │
        └── report ──→ Δ_eff / Δ_halluc / Δ_calib / Δ_cost + 配对符号检验
```

**agent 自报的数字一律不信**，全部重跑核对。没有这一步，测的就是「谁更会写报告」。

---

## 四、三个臂：把两层价值拆开

| 臂 | 追加的上下文（题面逐字相同） | 测的是 |
|---|---|---|
| **A** 裸 | `参考资料：无` | 基线 |
| **B** 库清单 | `+ libraries-registry.yaml`（有哪些库/版本/装没装） | 「知道在哪」 |
| **C** 完整 | `+ skill/`（atlas / scenarios / failure-modes / 判据） | 「知道怎么判断」 |

`B ≈ C` ⇒ 判据层没价值，只留库清单。

> 三臂只差「一行资源声明」，题面本身哈希校验必须一致。否则测的是提示长度，不是知识。

---

## 五、目录

```
toolkit-opbench/
├── README.md                本文件
├── NOTICE                   设计参考 NVIDIA/SOL-ExecBench (Apache-2.0)，未复制源码
├── LICENSE                  Apache-2.0
├── protocol.md              隔离 / GPU 互斥 / 评分 / 防作弊（含泄漏指纹表）
├── docs/
│   ├── architecture.md      三层架构与数据流（详版）
│   ├── design.md            为什么测增益、为什么不打外部榜
│   ├── format.md            题面三件套与 kind: judgment 规范
│   ├── taxonomy.md          9 家族 × 5 层
│   ├── scoring.md           两种分 + 四个治理指标 + Δ
│   └── workflow.md          六步工作流与常见故障表
├── bench/                   测量原语（见上）
├── harness/                 实验驱动（见上）
├── tasks/<family>-<op>-<tier>/
│   ├── task.md              人读题面（三臂逐字相同）
│   ├── definition.json      机器读（轴/张量/容差/expected_verdict）
│   ├── workload.jsonl       形状实例（含 holdout 与不规则形状）
│   ├── reference.py         参考实现（L1/L2）
│   ├── expect.md            参考答案（不给 agent）
│   └── check.py             题专属机检（可选）
├── suites/{l5-only,smoke,full}.yaml
├── baselines/<task>.json    T_b / T_sol 存档（跑一次存下来，评测只读）
├── runs/                    gitignored
└── reports/
```

---

## 六、六步工作流

```bash
python3 -m harness.doctor                 # 1 体检：GPU / 库版本 / 时钟 / 锁频能力 / L2
python3 -m harness.run --suite l5-only --arm a --prepare   # 2 展开队列
python3 -m harness.run --suite l5-only --arm a             # 3 执行（隔离 + 超时 + 收产物）
python3 -m bench.check.integrity --run runs/arm-a          # 4 验证（只认产物）
python3 -m bench.score.ratio --run runs/arm-a              # 5 计分
python3 -m harness.collect --runs runs --out reports/      # 6 出报告
```

常见故障与判定见 `docs/workflow.md`，其中三条是本项目踩过的：

| 现象 | 原因 |
|---|---|
| 带宽比峰值还高 | 没 flush L2（8 MB，实测出过 4571 GB/s） |
| 同一 kernel 两次差一倍 | 时钟档位不同（空闲 600 MHz / 工作档 1.15 GHz） |
| 重叠度两次结论相反 | 两侧耗时量级不匹配（必须 ∈ [0.2, 5]） |

---

## 七、DCU 适配要点（均为实测）

| 项 | 事实 | 设计要求 |
|---|---|---|
| 计时 | `torch.cuda.Event` 可用 | 禁 wall time，报 median + MAD |
| 时钟 | 空闲 **600 MHz**，工作档 **1.15 GHz** | 测量前 ramp 到工作档，每次记 `sclk` |
| 锁频 | `hy-smi` 无 `setclocks`，有 `--save/--load` | `doctor` 试「存→改→载」，失败只记录档位，**不假装锁了** |
| L2 | **8 MB** | flush buffer ≥ 16 MB，每轮迭代前 flush |
| 峰值 | fp8 **560 TF** / bf16 **310 TF** | 只引用 `bench/hw/constants.py` 的 id，**禁写字面量** |
| 多卡 | 8 卡，与其它会话共享 | 全局 `flock` + `hy-smi` 检测，串行执行 |

> 峰值分三类、不许混：`peak_nominal`（硬件标称，来自规格）/ `peak_achievable`（最佳实现实测）/ `impl_measured`（某库达成值，必须带 reproduce）。混用会同时出现「空间估大 10.9×」和「空间估小 10.9×」两种错，两种都发生过。

---

## 八、当前状态（2026-11-23 10:30）

| 模块 | 状态 |
|---|---|
| 目录 / 规范 / 题面模板 / suites | ✅ 已就位 |
| `bench/score/*`（ratio / sol_like / judgment / calibrate） | ✅ 已实现 + 单测 |
| `bench/data` · `bench/hw` · `bench/timing` · `bench/check` | 🚧 带签名的桩 |
| `harness/*`（doctor / run / collect / arms） | 🚧 带签名的桩 |
| 题库 | 1 道（L5）+ 模板；目标 39 道 |

桩不是空文件——每个都有 docstring 写清设计意图和踩过的坑，函数体 `raise NotImplementedError`。填桩时请先读注释。

---

## 九、与 SOL-ExecBench 的关系

借鉴了它的**题面三件套格式、反 reward-hack 检查、计时纪律（含 L2 flush、固定 warmup）**。
**未借鉴**：SOL-Score 的 B200 分母（gfx938 用不上）、「从零写 kernel」的题型（我们考选型与改法）、它的题目数据（许可与形状均不匹配）。

代码未复制，声明见 `NOTICE`。

---

## 十、怎么读这套文档

新手按这个顺序：`README` → `docs/design.md`（为什么测增益）→ `docs/architecture.md`（模块与数据流）→ `docs/format.md`（怎么写题）→ `docs/workflow.md`（怎么跑）。

只想知道「这套东西值不值」，直接看 `reports/`。
