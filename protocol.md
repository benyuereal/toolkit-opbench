# 操作协议

## 一、实验单元

一次实验 = `(task, arm)`。每个单元：

- 独立目录 `runs/<arm>/<task>/`，作为 agent 的 cwd
- 同一份 `tasks/<id>/task.md` 原样喂给三臂（**不许改一个字**）
- 固定预算：wall time（默认 1800s）+ token 上限
- 超预算的产出不计分

## 二、隔离（最关键，决定实验有效性）

三臂跑在**同一个容器、同一个用户**下，文件系统全可读。所以必须做隔离：

### 2.1 A 臂的隔离

跑 A 臂之前，把技能仓库移出 `/workspace`：

```bash
mkdir -p /var/.opbench-hidden
mv /workspace/toolkit_skills /var/.opbench-hidden/toolkit_skills
```

跑完立刻移回。**这是「防君子」不是「防小人」**，所以配第 2.2 条的检测。

### 2.2 泄漏指纹检测（必做）

A 臂如果偷看了 skill，它会说出**只有 skill 里才有的东西**。评分时扫 A 臂产出，命中任一条即判该题作废并标红：

| 指纹 | 说明 |
|---|---|
| `fuse_groupgemm_permute_quant` | 卡里的具体函数名 |
| `M* = 32-64` / `M/exp 32` | 只有 case 002 算出来的拐点 |
| `asm-fmoe-stage1-rejects-all-dtypes` | failure-modes 里的条目 id |
| `49.8 TF` / `560 TF` | 我们的标定结论 |
| `l1_exempt_reason` / `hard_preconditions` | 我们的 schema 字段名 |

发现泄漏 → 该题 A 臂结果作废，记入 `runs/leaks.log`，并在报告里注明。

**B 臂允许看到 `libraries-registry.yaml`**，但不许看到 atlas 卡、scenarios、failure-modes。

## 三、GPU 互斥（硬约束）

机器上还有一个长期会话 `claude-work` 在跑优化。**两个 agent 同时跑 bench 会互相污染。**

规则：
1. 跑 bench 前检查 `hy-smi` 的 HCU% —— 有任一卡 > 20% 就等
2. 用锁文件 `/tmp/opbench-gpu.lock`，`flock` 保护整个 bench 段
3. 单次 bench 里必须记 `clock_mhz` 与 `timestamp`（时钟漂移是我们踩过三次的坑）

## 四、评分（只认重跑）

### 4.1 每个单元判什么

| 项 | 判法 |
|---|---|
| **有效结论** | 产出里有「可复跑的数字 + 复跑方法」；缺 `results/` 判 0 |
| **正确** | 重跑通过 + 数值对（含**未提交 shape** + **压力数值**） |
| **加速比** | 跨 shape 几何平均，对基线 |
| **达到** | 宣称的结论是否被实测支持（校准度） |
| **幻觉** | 编造 `import_path` / 库名 / 行号：逐条 `import` + 文件存在性检查 |

### 4.2 幻觉的机检方法

```python
# 逐条验证 agent 声称的 import_path
mod, _, attr = claimed.rpartition('.')
m = importlib.import_module(mod)
assert hasattr(m, attr), f"编造: {claimed}"
```

库名对照 `libraries-registry.yaml`（canonical_id 白名单），不在表里 = 编造。

### 4.3 L5 反事实题的判分

L5 题目的正确答案是「**此路不通 + 机制**」。

| 回答 | 得分 |
|---|---|
| 「不可用，因为 X 报错 / 因为 Y 不支持」+ 有实测证据 | **100** |
| 「可用，预期能提升 Z%」/ 「应该可以，需进一步验证」 | **0**（幻觉） |
| 「不知道」 | 20（诚实但不完整） |

**L5 是唯一不跑 bench 的题，也是最便宜的一层。**

## 五、统计

- 配对设计：同题、同预算，三臂配对
- 样本量：每臂 5–10 题（单机现实），追求可重复而非大样本
- 检验：符号检验 / Wilcoxon signed-rank，报 p 值与效应量
- **先定胜负线再跑**，跑完不许改

## 六、报告格式

```markdown
## 结论一句话

## 三臂对比
| 题 | 臂 | 有效结论 | 正确 | 加速比 | 幻觉数 |
|---|---|---|---|---|---|
| l5-asm | A | ✗ | — | — | 1 |
| l5-asm | B | ✓ | ✓ | — | 0 |
| l5-asm | C | ✓ | ✓ | — | 0 |

## Δ
C−A 有效结论率: +XXpt (p=0.XX)
B−C 差距: XXpt（<10% ⇒ 判据层无价值）

## 泄漏与污染
（A 臂是否偷看、GPU 是否互斥、时钟是否记录）

## 下一步
```

## 七、不许做的事

1. 不许改 `task.md` 适配某一臂
2. 不许让评分器读 agent 的自述，只读产物
3. 不许在 GPU 被占用时跑 bench
4. 不许只报赢的题（全量报，包括 A 赢的题）
5. 不许在看到结果后改胜负线
