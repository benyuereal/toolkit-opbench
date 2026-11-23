# tookit-openeval-skills

> 评测「DCU 算子优化 skill」到底值不值。
> 不是跑分，是**测增益**：同等预算下，带 skill 的 agent 比裸 Claude Code 好多少。

## 这个仓库回答一个问题

> 我们那套 46 张卡 / 13 家族 / 6 场景的知识层，**提高了产出，还是只让文档变厚？**

**必须允许答案是「没提高」。** 如果 A 臂（裸）与 C 臂（完整 skill）分不出差别，
那该砍的是文档，不是继续加文档。

## 三个臂

| 臂 | 给什么 | 测的是 |
|---|---|---|
| **A** 裸 | 只给 `tasks/<id>/task.md` | 基线 |
| **B** 库清单 | + `libraries-registry.yaml`（有哪些库、版本、装没装） | 「知道在哪」 |
| **C** 完整 | + atlas / scenarios / failure-modes / 判据 | 「知道怎么判断」 |

**B≈C ⇒ 判据层没价值，只留库清单。**

## 怎么跑

```bash
set -a; source /workspace/opendesign.env; set +a
export PATH=/root/.nvm/versions/node/v24.22.3/bin:$PATH

# 单题单臂
bash run.sh --task l5-asm-fmoe-a8 --arm A --budget 1800

# 全量（5 题 × 3 臂）
bash run_all.sh

# 评分（重跑，不信自报）
python3 scorer/score.py runs/
```

## 结构

```
tasks/<id>/task.md      题目（三臂完全相同的 prompt）
tasks/<id>/expect.md    判分依据（不给 agent 看）
arms/                   A/B/C 的上下文注入脚本
runs/<arm>/<task>/      每臂每题一个目录，独立 cwd
scorer/                 评分器：重跑 + 机检幻觉 + 算 Δ
docs/design.md          评测设计（为什么这么测）
protocol.md             操作协议（隔离、互斥、防作弊）
```

## 三条铁律

1. **评分器不接受 agent 自报的数字** —— 必须重跑；`results/` 缺文件直接判 0
2. **A 臂不许看到 skill 仓库** —— 单容器下用「移走 + 指纹检测」，见 protocol.md
3. **GPU 不许两个 agent 同时跑 bench** —— 会互相污染，串行执行

## 胜负线（跑之前定死，跑完不许改）

```
C 相对 A 有效结论率 < 20%   ⇒ 砍文档，不是加文档
B 与 C 差 < 10%            ⇒ 判据层没价值，只留库清单
C 幻觉率 > A               ⇒ 立刻停，skill 在教模型编答案
```
