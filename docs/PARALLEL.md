# 并行写范围登记

> 并行 agent 开工前**先读本文件**，确认自己的写范围与别人不相交。
> 发现自己要写的路径已被别人占用 → 停下来说，不要覆盖。
>
> 最后更新：2026-11-23 14:05

## 总目标

对主要大模型算子做性能测试覆盖，让 toolkit-skills 的产出可被统一评测。

- **第一批（本轮验收线）**：gemm、rmsnorm
- 第二批（后续）：moe、attention、kv、quant、linear-attn

## 写范围（互不相交）

| 角色 | 写范围 | 状态 |
|---|---|---|
| **主线程** | `harness/**`、`suites/**`、`docs/**`、`README.md`、`reports/**` | 进行中 |
| **agent-gemm** | `tasks/gemm-bf16-l1/**`、`baselines/gemm-bf16-l1.json` | 待启动 |
| **agent-rmsnorm** | `tasks/norm-fused-add-rmsnorm-l1/**`、`baselines/norm-fused-add-rmsnorm-l1.json` | 待启动 |
| **agent-cli** | `bench/check/integrity.py`、`bench/score/*.py`、`Makefile`、`tests/test_cli.py` | 待启动 |

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
