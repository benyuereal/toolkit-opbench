# 怎么用这套 bench

> 一句话：`make ready` 绿了就能跑，`make cycle` 一条命令出结果。

---

## 一、开跑前：两个自检

```bash
make doctor    # 这台**机器**能不能测（GPU/库/时钟/锁频/L2）
make ready     # 这套**bench**能不能用（题目/基线/工具链/隔离）
```

两者都要绿（`!` 的 warn 项可以带着跑，但要记在报告里）。

`ready` 查 8 项：

| 项 | 查什么 | 红了怎么办 |
|---|---|---|
| task_schema | 16 道题的定义能加载 | 按报错改 definition.json |
| task_complete | kernel 六件套 / judgment 三件套 | 补齐缺的文件 |
| answer_isolated | expect.md 对 others 不可读 | `chmod 600 expect.md` |
| baseline | kernel 题有自洽基线 | `python3 -m harness.baseline --task <id>` |
| correctness | reference 自检恒等 | 修 reference.py |
| judge_scoring | judgment 打分能区分 100/20 | 查 verdict_keywords |
| tooling | 6 个工具入口可用 | 补 `__main__` |
| gpu | 卡数 / 时钟一致 / 有无空卡 | 等 claude-work 让卡 |

---

## 二、跑一次评测

### 最省事：一条命令

```bash
make cycle TASKS='mla-sparse-l1 moe-gating-l1'
```

它按顺序做：**建基线 → 跑三臂（A/B/C）→ verify → collect → 出报告**。
每步失败即停。

### 分步跑（想控制每一步时）

```bash
# 1. 建基线（一次性；已有则跳过）
python3 -m harness.baseline --task mla-sparse-l1

# 2. 跑一个臂
python3 -m harness.run --task mla-sparse-l1 --arm A --budget 1800

# 3. 跑一个 suite 的三臂
python3 -m harness.run --suite suites/smoke.yaml

# 4. 校验产物
python3 -m bench.check.integrity --run runs/arm-A

# 5. 汇总（必须给 --tasks，否则会把历史单元也算进来）
python3 -m harness.collect --tasks mla-sparse-l1 --print
```

### 跨卡并行（不同题用不同卡，同题三臂同卡）

```bash
python3 -m harness.parallel_run --tasks mla-sparse-l1 moe-gating-l1
```

**设计取舍**：同一道题的三臂**必须同一张卡**（否则 Δ 里混进卡间差异），
不同题可以并行。所以并行度 = min(题数, 空闲卡数)。

---

## 三、被测模型怎么配

三臂用**同一个模型**，否则测出来的是模型差不是 skill 差。

```bash
export OPBENCH_ENV=/workspace/tosc.env   # 端点与 token
export OPBENCH_EFFORT=medium             # qwen 只认 xhigh/medium/low
```

| 变量 | 作用 | 默认 |
|---|---|---|
| `OPBENCH_ENV` | 被测 agent 连 API 的环境文件 | `/workspace/opendesign.env` |
| `OPBENCH_EFFORT` | reasoning effort（空则不传） | 空 |
| `OPBENCH_MODEL` | 覆盖模型名 | 从 env 文件取 |
| `OPBENCH_DEVICE` | 绑卡（并行调度器设） | 自动挑空闲卡 |

---

## 四、三臂是什么

| 臂 | 给什么 | 测什么 |
|---|---|---|
| **A** 裸 | 只有题面 + 「参考资料：无」 | 基线能力 |
| **B** 库清单 | + `libraries-registry.yaml` | 「知道有哪些库」 |
| **C** 完整 skill | + `toolkit_skills`（只读软链） | 「知道怎么判断」 |

题面**逐字相同**（sha256 校验），只差最后一行资源声明。

---

## 五、结果怎么看

报告在 `reports/report.md`，三个维度：

### 性能（主指标，权重 80%）

| 数 | 含义 |
|---|---|
| `rel_ratio` | 本批最快 / 本臂 —— **横向排名**。谁最快谁 100% |
| `roof_ratio` | roofline 地板 / 本臂 —— **离物理极限多远** |
| `floor_kind` | `launch` 或 `bandwidth` —— 这个算子被什么支配 |

**只有 rel_ratio 不够**：第一名永远是 100%，不告诉你还有多少空间。
**只有 roof_ratio 也不够**：三个都很差时你不知道谁更好。

`floor_kind=launch` 意味着**优化 kernel 内部毫无意义**，该做融合或减 launch。

### 正确性（硬门槛）

不过 → 该单元总分 0，多快都没用。

整数输出（top-k 索引、token id）**必须精确相等**，不套浮点容差。

### 证据 / 幻觉

- 证据：有可复跑命令 + 结构化 `results/*.json`
- 幻觉：编造的 import_path / 库名 / 数字

---

## 六、测 skill 价值用哪套题

| suite | 用途 |
|---|---|
| `smoke` | 回归：证明 bench 没坏 |
| `l5-only` | 反事实题：敢不敢说"不通" |
| `skill-probe` | **对照实验**：证明 skill 有没有价值 |
| `full` | 全量 39 题（目前 12 道就绪） |

`skill-probe` 的选题原则：**答案不能靠 `dir()` 或读一个 API 文档得到**，
必须依赖「本机踩过的坑」或「判据」。这类知识只有 skill 里有。

例：`mla-sparse-l1` 的三个坑——topk 必须 128 倍数（跑才知道）、
-1 哨兵语义（skill 的 failure-modes 里有）、输出 512 不是 576。

---

## 七、加一道新题

```bash
mkdir tasks/<family>-<op>-<tier>
```

kernel 题要 6 个文件（`docs/format.md`）：
`task.md` / `definition.json` / `workload.jsonl` / `reference.py` / `check.py` / `expect.md`

```bash
chmod 600 tasks/<id>/expect.md        # 答案不给 A 臂看
chmod 640 tasks/<id>/definition.json  # 它含 expected_verdict
python3 -m harness.baseline --task <id>   # 建基线
make ready                                 # 确认没破规矩
```

judgment 题只要 3 个：`task.md` / `definition.json` / `expect.md`。

---

## 八、常见坑（都是实测踩过的）

| 现象 | 原因 | 处理 |
|---|---|---|
| `status=error` 且 `result_head` 有 402 | API 额度用尽 | 换 `OPBENCH_ENV` 或充值 |
| `reasoning effort high` 400 | 端点只认 medium/low | `OPBENCH_EFFORT=medium` |
| 达成率 >100% | 计时口径不同（burst 摊销 vs 逐次 event） | 用 `harness.measure` 统一测 |
| 同一实现三个臂报出三个数 | 各自口径不同 | 排名用 `harness.measure` 自己测 |
| 小算子谁都差不多 | 被 launch 地板支配 | 换大形状，或改题问"该不该做" |
| timeout 后进程还在跑 | `su` 建了新 session，killpg 杀不到 | 已修：按进程树 + cwd 兜底杀 |
| 报告里混进历史单元 | collect 扫了 runs/ 全部 | 必须传 `--tasks` |
