# toolkit-opbench · 项目设计

> 一句话：**在 DCU 上测「带 skill 的 agent 比裸 agent 好多少」，并且能判它是真做了还是编的。**

---

## 一、从 SOL-ExecBench 借鉴什么（借鉴，不复制）

已克隆 `NVIDIA/SOL-ExecBench`（Apache-2.0，13k 行 Python）看过内部结构。它的模块划分是：

```
core/bench/   correctness · timing · reward_hack · io
core/data/    definition · workload · solution · trace · shapes · dtypes
driver/       problem_packager · templates/{build_ext, eval_driver}
sol_score.py
```

### 借四条设计

| # | 它怎么做 | 我们怎么用 |
|---|---|---|
| 1 | **题面三件套**：`definition.json`（符号轴 + 张量规格 + 参考实现 + 容差）+ `workload.jsonl`（具体形状）+ `solution.json`（提交） | 直接采用结构。这让评分器能机器判，而不是读散文 |
| 2 | **反 reward-hack 做成一等公民**：检测 patch `torch.cuda.Event.elapsed_time`、后台线程注入、lazy tensor、评测函数被替换（id 快照） | 采用思路 + **加我们自己的一类**：编造 `import_path`、改题面、复用上次 results |
| 3 | **计时纪律**：CUPTI 取 GPU 时间（不是 wall time）、每轮 flush L2、warmup 10 / 计时 50–100、报 median + CV | 采用。DCU 上用 `torch.cuda.Event`，flush 用我们自己的 8 MB |
| 4 | **`benchmark_reference` 默认关闭**：参考实现可能跑 >1h，会主导总耗时 | 采用。我们的 `T_b` 单独跑一次并存档，评测时只读存档 |

### 不借三条

| 不借 | 原因 |
|---|---|
| **SOL-Score 的分母** | 它是 B200 的解析 roofline（`NVlabs/SOLAR`）。gfx938 一个数都用不上。我们用自己的 `constants.yaml`（fp8 560 / bf16 310 实测） |
| **"从零写 kernel"的题型** | 它的题产物是 kernel；我们是算子级选型 + 改法 + 判定。**L3/L5 它没有** |
| **它的题目数据** | 形状与硬件都不匹配，且 `reference.py`/description 有版权 |

### 许可

我们代码 Apache-2.0；根目录放 `NOTICE` 声明"设计参考 NVIDIA/SOL-ExecBench (Apache-2.0)"。**不复制其源码**。

---

## 二、架构：三层 + 一条数据流

```
┌─────────────────────────────────────────────────────┐
│  harness/   实验驱动（三臂、隔离、超时、收集）        │
│   跑的是"实验"，不是"一次测量"                        │
└───────────────────────┬─────────────────────────────┘
                        │ 调用
┌───────────────────────▼─────────────────────────────┐
│  bench/     测量原语（可 import 的库，不含实验逻辑）  │
│   data/   hw/   timing/   check/   score/            │
└───────────────────────┬─────────────────────────────┘
                        │ 读
┌───────────────────────▼─────────────────────────────┐
│  tasks/ + suites/ + baselines/   题、集合、基线存档   │
└─────────────────────────────────────────────────────┘
```

**为什么把 `bench` 和 `harness` 分开**：`bench` 是"怎么量得准"，`harness` 是"怎么比得公平"。混在一起，改实验设计时会动到测量代码，历史数据就不可比了。

### 数据流（一次完整评测）

```
task.definition + task.workload
        │
        ├─→ baselines/<id>.json          （T_b 与 T_sol，一次性存档）
        │
        ├─→ harness 为每个 arm 造 cwd ──→ claude -p ──→ solution + results/
        │                                                    │
        └────────────────────────────────────────────────────┘
                                                             ▼
                              bench.check: correctness / integrity / leak / hallucination
                                                             ▼
                              bench.timing: event 计时 + L2 flush + 时钟记录
                                                             ▼
                              bench.score: ratio / sol_like / judgment
                                                             ▼
                              harness.collect: 三臂 Δ ──→ reports/
```

---

## 三、目录结构

```
toolkit-opbench/
├── README.md
├── NOTICE                    设计参考 NVIDIA/SOL-ExecBench (Apache-2.0)
├── LICENSE                   Apache-2.0
├── docs/
│   ├── design.md             为什么测增益、三臂模型、为什么不打外部榜
│   ├── architecture.md       本文（模块与数据流）
│   ├── format.md             definition / workload / solution 规范
│   ├── taxonomy.md           9 家族 × 5 层
│   ├── scoring.md            两种分（ratio / sol_like）+ Δ
│   └── workflow.md           一次评测的 6 个步骤
├── bench/
│   ├── data/                 definition.py workload.py solution.py trace.py
│   ├── hw/                   dcu.py（探测/时钟/L2）+ constants.py
│   ├── timing/               event.py flush.py stats.py
│   ├── check/                correctness.py integrity.py leak.py hallucination.py
│   └── score/                ratio.py sol_like.py judgment.py calibrate.py
├── harness/
│   ├── run.py                单臂单题（隔离 / 超时 / 环境注入）
│   ├── arms/{a,b,c}.py       上下文注入策略
│   ├── collect.py            runs/ → reports/
│   └── doctor.py             体检：GPU / 库版本 / 时钟 / 锁频能力
├── tasks/
│   └── <family>-<op>-<tier>/
│       ├── task.md           人读（三臂逐字相同的 prompt）
│       ├── definition.json   机器读
│       ├── workload.jsonl   形状实例（含 holdout）
│       ├── reference.py     参考实现（L1/L2）
│       ├── expect.md        参考答案（不给 agent）
│       └── check.py         题专属机检（可选）
├── suites/                   l5-only.yaml  smoke.yaml  full.yaml
├── baselines/                每题的 T_b / T_sol 存档（JSON）
├── runs/                     gitignored
└── reports/
```

---

## 四、数据格式

### 4.1 题面三件套（沿用 SOL 的结构，字段我们自己定）

`definition.json`

```json
{
  "id": "gemm-bf16-l1",
  "kind": "kernel",
  "family": "gemm",
  "tier": "L1",
  "precision": ["bfloat16"],
  "axes": {
    "M": {"type": "var"},
    "N": {"type": "const", "value": 4096},
    "K": {"type": "const", "value": 4096}
  },
  "inputs": {"A": {"shape": ["M","K"], "dtype": "bfloat16"},
             "B": {"shape": ["K","N"], "dtype": "bfloat16"}},
  "outputs": {"C": {"shape": ["M","N"], "dtype": "bfloat16"}},
  "tolerance": {"rtol": 2e-2, "atol": 1e-2},
  "reference": "reference.py",
  "dcu": {
    "peak_constant": "fp8_peak_nominal_560",
    "requires_graph_capture": false,
    "min_l2_flush_mb": 16
  }
}
```

`workload.jsonl`（一行一个形状实例）

```json
{"axes": {"M": 4096}, "inputs": {"A": {"type": "random"}, "B": {"type": "random"}}, "uuid": "...", "split": "public"}
{"axes": {"M": 13312}, "inputs": {...}, "uuid": "...", "split": "holdout"}
```

`split: holdout` 的实例**题面里不出现**，只用于评分——这是留出集，防"只对着题面形状调参"。

### 4.2 judgment 变体（L3/L5 用）

```json
{
  "id": "moe-asm-fmoe-l5",
  "kind": "judgment",
  "family": "moe",
  "tier": "L5",
  "expected_verdict": "unavailable",
  "verdict_keywords": ["unsupport", "dtype", "mode", "namespace"],
  "forbidden_claims": ["可以融合", "预期提升", "建议上线"],
  "evidence_required": ["error_output", "command"]
}
```

`kind: judgment` 没有 `reference.py`。评分走"产物完整性 + 幻觉扫描 + expect 规则"，不做数值对比。

**这是它与 SOL-ExecBench 最大的结构差异**：它 235 题全是 `kind: kernel`，我们有 10 题是 `judgment`。

---

## 五、工作流：6 步

```bash
opbench doctor                          # 1 体检
opbench prepare --suite l5-only         # 2 展开 (task, arm) 队列
opbench run --suite l5-only             # 3 执行（隔离 + 超时）
opbench verify --suite l5-only          # 4 机检（正确性/完整/泄漏/幻觉）
opbench score --suite l5-only           # 5 算分与 Δ
opbench report --suite l5-only          # 6 出报告
```

### 第 1 步 doctor（每次开机跑一次）

| 检查 | 为什么 |
|---|---|
| 8 张卡可见 | 不可见就是环境没起对 |
| `torch.cuda.Event` 计时可用 | 计时原语 |
| **`hy-smi -g` 当前 sclk** | 空闲会降到 600 MHz；1.15 GHz 才是工作档 |
| **能否锁频**（`hy-smi --save/--load`） | 不能锁就要记录实际档位并放进统计 |
| L2 = 8 MB | flush buffer 大小 |
| 库版本：lightop / aiter / deepgemm / torch / DTK | 不记录就不可复跑 |
| 另一会话是否占卡 | GPU 互斥 |

### 第 3 步 run（每单元的执行契约）

```
1. 建 runs/<arm>/<task>/ 作为 cwd，chmod 777（A 臂要用 opeval 用户写）
2. 注入上下文（A 无 / B 只给 libraries-registry.yaml / C 给 skill 仓库）
3. 预热时钟（跑一段 dummy 直到 sclk 到工作档），记录 sclk
4. timeout <budget> claude -p "$(cat task.md)" --output-format json
5. 产物落盘：report.md / scripts/ / results/
6. 记录：wall time、token、sclk、exit code、GPU 锁释放
```

**超时即判失败**，不续跑——补跑会破坏固定预算的可比性。

### 第 4 步 verify（只认产物）

| 检查 | 判法 |
|---|---|
| correctness | 重跑 reference，逐 shape 对比（含 holdout + 压力数值） |
| integrity | task.md 哈希未变；评测函数未被替换；无后台线程；无 lazy tensor |
| leak | A 臂产物扫 6 条指纹（命中即作废该题） |
| hallucination | 声称的每个 `import_path` 逐条 `importlib` + `hasattr` |
| artifact | `results/` 存在且字段齐全，缺则判 0 |

### 第 5 步 score

见下节。

---

## 六、DCU 适配清单（这节是"结合本地海光"的核心）

| 项 | 事实（已实测） | 设计动作 |
|---|---|---|
| 计时 | `torch.cuda.Event` ✅ 可用 | 计时**只用** event，禁止 wall time；报 median + MAD |
| **时钟** | 空闲 **600 MHz**，工作档 1.15 GHz | 测量前 ramp；每次测量记 `sclk`；报告里注明档位 |
| 锁频 | `hy-smi` 无 `--setclocks`，但有 `--save/--load` | doctor 试一次"存→改→载"；失败则记录档位，不假装锁了 |
| **L2** | **8 MB** | flush buffer ≥ 16 MB（2×），每轮迭代前 flush，否则热数据虚高（我们踩过 4571 GB/s） |
| 峰值分母 | fp8 560（业主确认）· bf16 310（实测） | `constants.yaml` 三类分开，任务里只引用 id，不写字面量 |
| 库生态 | lightop / aiter / deepgemm / flash_mla / tilelang | 每题记录实际库与版本；这是复跑的前提 |
| 图捕获 | 部分 kernel 只有走 graph 才接近稳态 | `definition.dcu.requires_graph_capture` 显式标注 |
| 多卡 | 8 卡，另一会话可能占用 | 全局 `/tmp/opbench-gpu.lock` + `hy-smi` 占用检测，**串行** |

---

## 七、评分：两种分，用途不同

### 7.1 `ratio`（主指标，用于奖励与比较）

```
ratio = T_b / T_k        # 对基线的加速比，越大越好，不饱和
```

### 7.2 `sol_like`（可读分，用于报告）

借鉴 SOL 的形式：

```
S = 1 / (1 + (T_k - T_sol) / (T_b - T_sol))
  T_sol = 我们的 roofline 地板（用 constants.yaml + models/）
  T_b   = 基线（基线的 S = 0.5，光速时 S = 1）
```

**关键：`sol_like` 只用于报告，不用于奖励。** 它在接近光速处会饱和——CompoundingKernel 就是因为这个不用 SOL 分做奖励。我们两种都算，但只用 `ratio` 比高下。

### 7.3 `judgment`（L3/L5）

| 回答 | 分 |
|---|---|
| 结论正确 + 有实测证据 + 说清机制 | 100 |
| 结论正确但证据不足 | 60 |
| "不知道 / 未验证" | 20（诚实） |
| 结论错误 / 编造证据 | 0，并计入幻觉数 |

### 7.4 三臂 Δ（最终要的那一个数）

```
Δ_eff      = C.有效结论率 − A.有效结论率
Δ_halluc   = C.幻觉率 − A.幻觉率        # 期望 ≤ 0
Δ_calib    = |宣称达成率 − 实测达成率|  # ≤ 10pt
```

**胜负线（跑前定死）**：`Δ_eff < 20pt` ⇒ 砍文档；`B 与 C 差 < 10pt` ⇒ 判据层无价值；`Δ_halluc > 0` ⇒ 立刻停。

---

## 八、里程碑

| 阶段 | 内容 | 交付 |
|---|---|---|
| **M0** 半天 | `bench/data` + `bench/hw` + `doctor.py`；能从 definition+workload 造输入并跑通一个 reference | `opbench doctor` 全绿 |
| **M1** 一天 | `bench/timing`（event + 8MB flush + 时钟记录）+ `bench/check/integrity` | 3 道 L1 题能出 ratio |
| **M2** 一天 | `harness`（三臂 + 隔离 + 超时）+ `collect` + `scorer` | `l5-only` 出一份 Δ |
| **M3** 2–3 天 | 题库扩到 39 + 三套 suite + 报告模板 | 正式报告 |

---

## 九、风险（提前写下来）

1. **锁频可能做不到** → 退路：记录档位 + 增加重复次数 + 在报告里标"未锁频"，结论里降置信度
2. **A 臂隔离是"防君子"** → 退路：泄漏指纹检测 + 报告声明局限
3. **L3/L4 需要真实仓库环境**，可能污染被测系统 → 退路：L4 只在独立 worktree 里跑
4. **题库 39 题工作量大于预期** → 退路：先 `l5-only` 出结论，题库按家族渐进
