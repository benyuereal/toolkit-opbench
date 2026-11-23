# 并行写范围登记

> 并行 agent 开工前**先读本文件**，确认自己的写范围与别人不相交。
> 发现自己要写的路径已被别人占用 → 停下来说，不要覆盖。
>
> 最后更新：2026-11-23 15:55

## 总目标

对主要大模型算子做性能测试覆盖，让 toolkit-skills 的产出可被统一评测。

- **第一批（本轮验收线）**：gemm、rmsnorm
- 第二批（后续）：moe、attention、kv、quant、linear-attn

## 写范围（互不相交）

| 角色 | 写范围 | 状态 |
|---|---|---|
| **主线程** | `harness/**`、`suites/**`、`docs/**`、`README.md`、`reports/**`、`bench/**` | 进行中 |
| **agent-attn** | `tasks/attn-paged-decode-l1/**`、`baselines/attn-paged-decode-l1.json` | 启动 |
| **agent-kv** | `tasks/kv-paged-write-l1/**`、`baselines/kv-paged-write-l1.json` | 启动 |
| **agent-linear** | `tasks/lin-causal-conv1d-l1/**`、`baselines/lin-causal-conv1d-l1.json` | 启动 |
| **agent-head** | `tasks/head-lm-sampling-l1/**`、`baselines/head-lm-sampling-l1.json` | 启动 |
| ~~agent-gemm~~ | 已由主线程完成（子 agent 因网关 524 超时终止） | ✅ 完成 |
| ~~agent-rmsnorm~~ | 已由主线程完成（同上） | ✅ 完成 |
| ~~agent-cli~~ | 已由主线程完成 | ✅ 完成 |

**禁止跨范围写入。** 需要改别人的文件 → 停下来报告，由主线程协调。

## 共享只读资源

以下文件**任何 agent 都不要改**，只能读：

- `bench/hw/constants.py` — 峰值常数的唯一来源（三类口径）
- `bench/data/**` — 题面 schema
- `bench/timing/**` — 计时原语
- `tests/` 下除 `test_cli.py` 之外的既有测试

## GPU 互斥（硬约束）

只有一个 DCU，且 `claude-work` 会话也在跑。

- **纯 CPU 的活**（写题面、补 CLI 入口、跑单测）不受限
- **跑 bench**（任何 `torch.cuda` 计算）必须：
  1. 持 `/tmp/opbench-gpu.lock`（`bench.hw.dcu.gpu_lock()`）
  2. 持锁后**再查一次** `hy-smi -u`，任一卡 > 20% 就等
  3. 测量前 ramp 到工作档、记录 sclk
- 三个 agent 同时跑 bench 会互相污染 —— 由主线程串行调度

## 时间线

| 时间 | 事件 |
|---|---|
| 14:05 | 建立本文件；主线程开始修 verify/score |
| 14:40 | 两个子 agent 均因网关 524 超时终止，未留下任何文件；改为主线程完成 |
| 15:20 | verify/score 真入口 + 退出码测试完成；评分器区分度修复 |
| 15:50 | gemm / rmsnorm 六件套完成；带宽口径 + 3 处计时/检测 bug 修复 |
