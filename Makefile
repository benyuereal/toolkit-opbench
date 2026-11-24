# toolkit-opbench 常用命令
#
# 六步工作流见 docs/workflow.md；这里只是把长命令收短。
# 无桩：bench/* 与 harness/* 全部已实现，222 个单测见 tests/。

PY      ?= python3
SUITE   ?= l5-only
ARM     ?= c
RUNS    ?= runs
OUT     ?= reports

.PHONY: help test doctor prepare run verify score report check-env clean cycle

help:
	@echo "make test                 跑单测（无需 GPU）"
	@echo "make doctor               环境体检（GPU/库/时钟/锁频/L2）"
	@echo "make ready                bench 自检（题目/基线/工具链）"
	@echo "make prepare SUITE=x      展开 (task, arm) 队列"
	@echo "make run SUITE=x ARM=c    执行一个臂"
	@echo "make verify RUNS=runs     验证产物（只认产物，不信自报）"
	@echo "make score RUNS=runs      计分"
	@echo "make report               汇总成报告"
	@echo "make cycle [TASKS='...']  完整周期（基线→三臂→校验→汇总）"
	@echo "make check-env            打印被测版本与依赖快照"

test:
	$(PY) -m pytest tests/ -q

doctor:
	$(PY) -m harness.doctor

# bench 自检：题目 / 基线 / 工具链 是否就绪（不开跑）
ready:
	$(PY) -m harness.ready

prepare:
	$(PY) -m harness.run --suite $(SUITE) --arm $(ARM) --prepare

run:
	$(PY) -m harness.run --suite $(SUITE) --arm $(ARM)

verify:
	$(PY) -m bench.check.integrity --run $(RUNS)/arm-$(ARM)

score:
	$(PY) -m bench.score.ratio --run $(RUNS)/arm-$(ARM)

report:
	$(PY) -m harness.collect --runs $(RUNS) --out $(OUT)

# 完整周期：基线 → 三臂 → 校验 → 汇总，一道命令走完
TASKS ?= gemm-bf16-l1 norm-fused-add-rmsnorm-l1 moe-gating-l1
cycle:
	$(PY) -m harness.cycle --tasks $(TASKS)

check-env:
	@echo "=== 本仓库 ==="; git rev-parse --short HEAD
	@echo "=== 被测 skill 仓库 ==="; git -C ../toolkit-skills rev-parse --short HEAD 2>/dev/null || echo "（未挂载）"
	@echo "=== torch ==="; $(PY) -c "import torch; print(torch.__version__, 'hip=', torch.version.hip)" 2>/dev/null || echo "无 torch"
	@echo "=== 时钟 ==="; hy-smi -g 2>/dev/null | tail -3 || echo "无 hy-smi"

clean:
	rm -rf .pytest_cache **/__pycache__
