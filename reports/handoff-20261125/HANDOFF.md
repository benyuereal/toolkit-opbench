# HANDOFF — opbench 三臂对比实验

> 交接时间：2026-11-25 09:10（CST）
> 交接人：前一任执行者（AI 助手）
> 接手人需要：能 SSH、会用 tmux、读过 `CLAUDE.md`

---

## 1. 这件事到底在测什么

**一句话**：验证"给 Claude Code 挂上算子知识库（skill）后，跑 GPU kernel 优化任务的效果是否比裸 Claude Code 更好"。

**三臂设计**（`suites/three-way.yaml`）：

| 臂 | 含义 | 用户 | 环境 |
|---|---|---|---|
| **A** | 裸 Claude Code | `opb_a` | 只有题面 |
| **C** | toolkit-skills + CC | `opb_c` | 挂 `/workspace/toolkit_skills`（含 `loop.py` 迭代驱动） |
| **D** | SkillHub + CC | `opb_d` | 挂 `/workspace/SkillHub`（44 个团队共享 skill） |

**关键**：三臂的 prompt **逐字相同**，差异只在"环境里有没有那套 skill"。这样测出来的差才是 skill 的贡献。

---

## 2. 环境

### 2.1 怎么连

```bash
# 宿主机（免密，root）
ssh root@10.20.100.39

# 容器（实际干活的地方）
docker exec -it glm53-flash-yuebin bash
```

**路径映射（重要）**：

```
宿主机 /data/yuebin/  ══ bind mount ══  容器 /workspace/
项目：/workspace/toolkit-opbench
```

改文件在宿主机改，容器里立即可见；反之亦然。

### 2.2 模型端点

```
env 文件：/workspace/opendesign-sk.env   （chmod 644，所有臂用户可读）
端点    ：https://amr-link.open-design.ai
模型    ：deepseek-v4.1-flash
```

其他可用端点（已实测 2026-11-25）：
- `opendesign-benyuereal.env` → 200，同模型
- `tosc.env` → qwen3.8-27b，内网 `10.20.100.15:4001`，慢但稳
- 已 402 的别用：`opendesign.env`、`opendesign-benglobal.env`

### 2.3 GPU

8 张 DCU（gfx938 / BW1100），`hy-smi` 看状态。三臂并行时各绑一张（dev 0/1/2）。

---

## 3. 本轮中间结果（⚠️ **仍在跑，不是终值**）

> **重要**：截至 09:14，三臂 harness 进程都还在运行（08:47:37 启动，budget 3600s，预计 09:47 结束）。
> 下面的数字是**中途写出的 `bench_std.json` 快照**，agent 可能还在继续迭代，最终值会变。
> 判断依据：attempt 目录里**没有 `meta.json`**（harness 只在结束时写），也**没有 `report.md`**。

**题目**：`norm-fused-add-rmsnorm-l1`，budget 3600s
**口径**：4 趟（2R2W = 写 y + 写 h），baseline = `lightop.fused_add_rms_norm`
**题面约束**：禁止使用 `lightop`，60 分钟预算

### 3.1 中间快照（09:10 左右）

| 臂 | 当前实现 | M=4096 (ms) | M=8192 (ms) | ratio 4096/8192 | geomean |
|---|---|---|---|---|---|
| **A 裸 CC** | **`triton_fused_rpm1_w8`（自己写的 triton kernel）** | 0.088 | 0.17424 | 1.287 / 1.238 | **1.262x** |
| D SkillHub | `aiter.ops.triton.rmsnorm2d_fwd_with_add` | 0.12264 | 0.20560 | 0.924 / 1.049 | 0.984x |
| C toolkit | `aiter.fused_add_rms_norm_cu`（原地） | 0.23192 | 0.44288 | 0.488 / 0.487 | 0.488x |

baseline 参考：`0.11328 / 0.21568` ms

### 3.2 A 的 1.262x 已验证合法

`scripts/bench_final.py` 里 kernel 有**两个 `tl.store`**：

```python
tl.store(r_ptr + off, h.to(tl.bfloat16), mask=mask)              # 写 h（residual，传给下一层）
tl.store(x_ptr + off, (h * rstd * w).to(tl.bfloat16), mask=mask) # 写 y（本层输出）
# 注释原文：「流量：2 读 + 2 写 = 4*M*H*2 字节」
```

→ 与 baseline 同为 4 趟，**不是偷工减料**。

`achieved_pct_median = 113%` 的原因：分母用了 copy 峰值 `1350 GB/s`，
而实测 `1525 GB/s`（read 峰值 1965 的 78%）。读写混合超过纯 copy 峰值是正常的。

A 还自己扫了参数：`all_candidates` 里有 `triton_fused_rpm1_w4`(0.08968)、`rpm1_w8`(0.088) 等。

### 3.3 结论：裸 CC 完胜，与预期相反

| 假设 | 实际 |
|---|---|
| 挂 skill 的 C/D 应该更强 | **A（裸 CC）1.262x >> D 0.984x > C 0.488x** |
| C 有 loop.py 强制迭代，应该最好 | **C 最差**，且比上一轮（0.90x）还退步 |
| 禁用 lightop 会让三臂分化 | 分化了，但方向是"裸 CC 自研 kernel" |

**最可能的原因：skill 把 agent 引导向"找现成实现"，抑制了自研。**

- **A**：没有任何 skill 提示，看到"要优化这个算子"，直接写 triton kernel
- **D**：44 个团队 skill 里全是"用 aiter / 用库"的经验，它就去找 `aiter.ops.triton.*`
- **C**：toolkit-skills 指向 `aiter.fused_add_rms_norm_cu`，**还挑了个原地版本（最慢）**

> **反向证据**：算子知识库里的"最佳实践"可能固化了思维，反而挡住了"自己写一个"这条路。

### 3.4 上一轮（已作废）对照

用原题面（30 min，允许 lightop）：
- A/D 都找到 `lightop.op.fused_add_rms_norm_opt`，**1.11x**
- C 审慎选了 `aiter.rmsnorm2d_fwd_with_add`（非原地），0.90x，loop 只跑 1.5 轮就超时

**为什么作废**：三臂趋同（都靠翻 lightop），测不出 skill 差异 → 改题面禁用 lightop 重跑。

### 3.5 C 的 loop 流水（本轮）

C 确实被强制调用了 `loop.py`（这一点上一轮是完全没调），产物在
`/tmp/opbench-run/arm-C/norm-fused-add-rmsnorm-l1/attempt-1/skill/Plugins/ops/skills/op-fused-add-rmsnorm/runs/loop/*.jsonl`
（上一轮典型内容：`aiter-fused-add` → `REJECTED_NO_GAIN` / `REJECTED_ACCURACY`）

**但要注意**：C 的分数反而比"不走 loop"时更低。**loop 的自我否证机制可能是双刃剑**——
它否掉了噪声内的候选（正确），但也可能让 agent 放弃了本来能赢的方向。

---

## 4. 今天做过的关键改动（都在 git 里，未提交）

### 4.1 测量口径统一（`harness/measure_spec.py` 新增）

**问题**：题面只有张量语义（`[S,H,576]`），没给具体 axes。实测 A 臂自选 `SKV=16384/TOPK=2048`，
C/D 用 `8192/256`，S=512 时 **2542 μs vs 163 μs** —— 差 15 倍，根本不是同一个算子。

**修法**：harness 从 `baselines/<task>.json` 读官方 shapes，生成"测量口径"段落，
**注入到三臂的 prompt 末尾（逐字相同）**，并要求结果写成 `results/bench_std.json`。

```python
# harness/run.py 的 run_one()
prompt = mod.build_prompt(task_md)
from harness import measure_spec
_spec = measure_spec.render(task_id)
if _spec:
    prompt = prompt.rstrip() + _spec
```

### 4.2 强制 C 臂走 loop.py

**问题**：C 臂第一轮 **调用 loop.py 的次数是 0** —— 它读了 SKILL.md 就去写代码了，
"强制迭代"这个核心假设根本没验证。

**修法**（两手）：
1. `harness/arms/c.py` 的 `CLAUDE_MD` 加「完成标准（硬要求）」：
   > 必须执行过 `loop.py -C <卡> run <tag>`，卡目录下必须留下 `runs/loop/*.jsonl`。
   > **没有这个产物，本次交付直接判 `invalid:no_loop_evidence`。**
2. `harness/run.py` 加校验：
   ```python
   def _requires_loop(arm): return arm.upper() == "C"
   def _loop_evidence(cwd): return glob(cwd/**/runs/loop/*.jsonl, recursive=True)
   # _execute 成功后
   ev = _loop_evidence(cwd)
   if _requires_loop(arm) and not ev:
       return "invalid:no_loop_evidence", ...
   ```

**效果**：C 第二轮立刻调了 `loop.py status`，并产出了 `runs/loop/20261125.jsonl`。

### 4.3 一臂一用户隔离（最重要）

**问题链**（每修一个暴露下一个）：
1. A 臂 root 跑 → 能读 `expect.md`（答案）→ 改 `USER = "opbench"`
2. 三臂都用 opbench → **C 读了 A 的 report.md**（实测 `sed -n 1,200p /tmp/opbench-run/arm-A.pre-2340/.../report.md`）
3. → 改成**一臂一用户**：`opb_a` / `opb_c` / `opb_d`

**实现**：
- `harness/arms/{a,c,skillhub}.py` 的 `USER` 分别指向三个新用户
- `harness/run.py` 新增 `_claim_for_arm(arm, path)`：`chown` + `chmod 700` 整棵 `arm-X` 树
- `c.py` / `skillhub.py` 的 `stage_files` 末尾 `chown -R <user>:<user>` + `chmod -R u+rwX,go-rwx`

**当前隔离矩阵**：

| 资源 | 谁能读 |
|---|---|
| `tasks/*/expect.md` (600 root) | 仅 root |
| `baselines/`、`runs/` (700 root) | 仅 root |
| `/tmp/opbench-run/arm-X` (700 各臂用户) | 仅该臂用户 |
| `task.md` / 自己的 skill 副本 | 各自臂 |

**已验证**：`C_BLOCKED_A`、`A_BLOCKED_C`、`opb_c` 读 `baselines/` 被拒。

### 4.4 题面改动（这一轮生效）

`tasks/norm-fused-add-rmsnorm-l1/task.md`（备份在 `/tmp/task_norm_fused.md.bak`）：

```diff
- 时间预算：**30 分钟**
+ 时间预算：**60 分钟**

+ - **禁止使用 `lightop` 库中的算子**（`lightop.fused_add_rms_norm`、`lightop.op.*`、
+   `lightop.RMSNormAdd`、`lightop.norm.*` 一律不可作为落点，也不可用作候选）。
+   这一条是硬约束：这条链路上 lightop 是待替换对象，不是可用工具。
+   你要在**其余可用实现**里找，或自己写。
```

**为什么禁用 lightop**：上一轮 A/D 都靠翻 lightop 找到 `_opt` 变体，殊途同归，测不出 skill 差异。

---

## 5. 踩过的坑（别再踩）

| 坑 | 现象 | 解法 |
|---|---|---|
| `docker exec -d` 启动即退 | 日志 0 字节、无进程 | 改用 `tmux new-session -d` |
| 单题模式中途无输出 | 以为卡死 | `claude -p --format json` 只在结束时输出，看 `meta.json` 或 session 文件 |
| A 臂 env 权限 | `Permission denied`，39 秒秒退 | env 文件必须 `644`（A 臂是普通用户） |
| `%/*` 去斜杠 | `https://x.ai` → `https:/` → `ENOTFOUND` | 用 `${VAR%/}` |
| cwd 缺 `scripts/` `results/` | Write 失败 → agent 反复查权限跑偏 | `isolated_cwd` 里预建这两个目录 |
| 验证权限时 `rm -rf` 了正在跑的 cwd | 所有命令 EACCES（死 inode） | **别动正在运行的 cwd** |
| 绑卡分支漏传 `plugin_dir` | 并行时 C/D 的 skill 加载不上 | `_execute(..., plugin_dir=plugin_dir)` |
| `su -` 丢环境 | `HIP_VISIBLE_DEVICES` 没传进去 | 在子 shell 里显式 `export` |
| `.py` 被判 sensitive file | Write/Edit 被拒 | 已加 `--dangerously-skip-permissions`（隔离边界靠 Unix 权限） |

---

## 6. 常用命令

```bash
# 跑单题单臂
docker exec -w /workspace/toolkit-opbench -e OPBENCH_ENV=/workspace/opendesign-sk.env \
  glm53-flash-yuebin python3 -u -m harness.run \
  --task norm-fused-add-rmsnorm-l1 --arm C --budget 3600 --max-usd 5.0

# 三臂并行（现成脚本）
docker exec glm53-flash-yuebin bash /tmp/run_verify.sh A 0 norm-fused-add-rmsnorm-l1
docker exec glm53-flash-yuebin bash /tmp/run_verify.sh C 1 norm-fused-add-rmsnorm-l1
docker exec glm53-flash-yuebin bash /tmp/run_verify.sh D 2 norm-fused-add-rmsnorm-l1

# 出三臂性能表（只认 bench_std.json，会校验 axes）
docker exec -w /workspace/toolkit-opbench glm53-flash-yuebin python3 tools/report_std.py

# 看 C 的实时轨迹（loop.py 调用、工具动作）
docker exec -w /workspace/toolkit-opbench glm53-flash-yuebin python3 /tmp/dump_trace.py
```

**注意**：C/D 的 cwd 现在在 `/tmp/opbench-run/arm-X/`（不是 `runs/arm-X/`），因为一臂一用户隔离。
`tools/report_std.py` 已经适配了双路径。

---

## 7. 待办 / 已知问题

### 7.1 下一步要做的（按优先级）

1. **把干净口径套到另外两题**：`moe-gating-l1` / `mla-sparse-l1` 也要改题面（禁 lightop + 60min）重跑。
   否则 3 道 L1 里只有 norm-fused 这 1 道可信。
2. **跑 L5 六道反事实题**（judgment 模式，无 t_sol）—— 完全没跑过。
3. **查 A 为什么能自研成功**：
   - A 的 attempt 目录里**没有 `report.md`**（只有 `scripts/` 和 `results/`），先确认它是没写还是没写完
   - 看 A 的 session 轨迹（`/home/opb_a/.claude/projects/*arm-A-norm-fused*/*.jsonl`），
     搞清楚"裸 CC 是怎么决定写 kernel 而不是找库"的。**这决定了 skill 该怎么改**。
4. **查 C 的 loop 为什么选了最差的实现**：
   - C 本轮用 `aiter.fused_add_rms_norm_cu`（原地，0.488x）
   - 但它上一轮**明确说过**"`aiter.fused_add_rms_norm_cu` 原地比显式 out 慢 1.84x，不作为候选"
   - 要查是 loop 的候选排序把它推过去了，还是它自己判断反转了

### 7.2 结果文件在哪

```bash
# 三臂的完整产物（A/C/D 各一份）
/tmp/opbench-run/arm-A/norm-fused-add-rmsnorm-l1/attempt-1/
/tmp/opbench-run/arm-C/norm-fused-add-rmsnorm-l1/attempt-1/
/tmp/opbench-run/arm-D/norm-fused-add-rmsnorm-l1/attempt-1/
# 每份含：bench_std.json（性能）、meta.json（状态/用量）、scripts/（脚本）、report.md（A 没有）

# C 的 loop 流水
/tmp/opbench-run/arm-C/norm-fused-add-rmsnorm-l1/attempt-1/skill/Plugins/ops/skills/op-fused-add-rmsnorm/runs/loop/*.jsonl
```

### 7.2 结构性问题

| 问题 | 说明 | 建议 |
|---|---|---|
| **baseline 与题面矛盾** | 题面说"当前实现是未融合三段式"，但 `baselines/` 里 `baseline_impl = lightop.fused_add_rms_norm`（已经是融合版） | 要么改题面，要么重测 baseline |
| **`t_b_ms` 是几何平均** | `t_b` = 两档 median 的几何平均（`sqrt(0.2627*0.5114)`），**不能当单档分母** | `tools/report_std.py` 已改为按 uuid 逐档匹配 |
| **mla-sparse 的 baseline 可疑** | baseline 记 S512=0.2627，但三臂实测都是 0.128~0.132（同一实现） | 该 baseline 需重测，否则 ratio 全虚高 |
| **`harness/` 可被臂读** | 755 root，agent 能读 `run.py`（含判分逻辑） | 建议 `chmod 700 harness`（未做，怕影响运行） |
| **D 臂没有 loop 硬校验** | SkillHub 的 loop 是 skill 形态，无统一产物路径 | 可校验 `_stdout.jsonl` 里是否加载过 `*agent-loop*` skill |

### 7.3 实验设计层面

- **只有 norm-fused 这一题真正验证了**。mla-sparse / moe-gating 用的是旧题面（30min、允许 lightop），结论已作废。
- **`suites/three-way.yaml` 里的 3 道 L1 题**（moe-gating / norm-fused / mla-sparse）应统一改成新口径后全部重跑。
- **L5 六道反事实题**（judgment 模式，无 t_sol）完全没跑过。

---

## 8. 目录速查

```
/workspace/toolkit-opbench/
├── harness/
│   ├── run.py            # 主入口（改了很多：隔离、口径注入、loop 校验）
│   ├── measure_spec.py   # 新增：测量口径生成
│   └── arms/{a,c,skillhub}.py  # 三臂定义（USER、stage_files）
├── tasks/norm-fused-add-rmsnorm-l1/
│   ├── task.md           # 题面（已改：60min + 禁 lightop）
│   ├── expect.md         # 答案（600 root）
│   └── definition.json
├── baselines/*.json      # 基线数据（700 root）
├── tools/report_std.py   # 三臂性能报表（按 uuid 逐档匹配）
├── tests/test_measure_spec.py  # 口径守卫测试（6 项）
└── runs/                 # 历史结果（700 root，含 .pre-2340 备份）
```

**运行产物**：`/tmp/opbench-run/arm-{A,C,D}/<task>/attempt-1/`
（含 `report.md`、`scripts/`、`results/bench_std.json`、`meta.json`）

---

## 9. 一句话总结现状

> **机制搭好了（一臂一用户隔离 / 强制 loop.py / 测量口径统一），第一道题（norm-fused）正在
> 用干净口径跑第一次公平比较，09:47 出终值。**
>
> **中途快照（09:10）显示一个反直觉的信号：裸 CC 自己写 triton kernel（1.262x）远好于
> D（0.984x）和 C（0.488x）。但三臂都还在跑，这个顺序随时可能变，不能当结论用。**
>
> 等终值出来后再判断"skill 是帮了还是害了"。另外两道 L1（moe-gating / mla-sparse）
> 和 6 道 L5 都还没用新口径跑过。
