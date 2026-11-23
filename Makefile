# toolkit-opbench 常用命令
#
# 六步工作流见 docs/workflow.md；这里只是把长命令收短。
# 注意：harness/* 与 bench/{data,hw,timing,check} 目前是桩，跑会 raise NotImplementedError。

PY      ?= python3
SUITE   ?= l5-only
ARM     ?= c
RUNS    ?= runs
OUT     ?= reports

.PHONY: help test doctor prepare run verify score report check-env clean

help:
	@echo "make test                 跑单测（无需 GPU）"
	@echo "make doctor               环境体检（GPU/库/时钟/锁频/L2）"
	@echo "make prepare SUITE=x      展开 (task, arm) 队列"
	@echo "make run SUITE=x ARM=c    执行一个臂"
	@echo "make verify RUNS=runs     验证产物（只认产物，不信自报）"
	@echo "make score RUNS=runs      计分"
	@echo "make report               汇总成报告"
	@echo "make check-env            打印被测版本与依赖快照"

test:
	$(PY) -m pytest tests/ -q

doctor:
	$(PY) -m harness.doctor

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

check-env:
	@echo "=== 本仓库 ==="; git rev-parse --short HEAD
	@echo "=== 被测 skill 仓库 ==="; git -C ../toolkit-skills rev-parse --short HEAD 2>/dev/null || echo "（未挂载）"
	@echo "=== torch ==="; $(PY) -c "import torch; print(torch.__version__, 'hip=', torch.version.hip)" 2>/dev/null || echo "无 torch"
	@echo "=== 时钟 ==="; hy-smi -g 2>/dev/null | tail -3 || echo "无 hy-smi"

clean:
	rm -rf .pytest_cache **/__pycache__
