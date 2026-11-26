#!/bin/bash
# 单臂执行器：一臂一卡，三臂并行。
# 用法: run_arm.sh <ARM> <DEVICE>
ARM=$1; DEV=$2
cd /workspace/toolkit-opbench || exit 1
export OPBENCH_ENV=/workspace/opendesign-sk.env
export OPBENCH_DEVICE=$DEV
LOG="runs/par_${ARM}_$(date +%m%d_%H%M).log"
say(){ echo "$*" >> "$LOG"; }
stdpath(){ if [ "$ARM" = "A" ]; then echo "/tmp/opbench-run/arm-A/$1/attempt-1"; else echo "runs/arm-$ARM/$1/attempt-1"; fi; }
say "##### ARM=$ARM DEV=$DEV START $(date) #####"
for t in mla-sparse-l1 norm-fused-add-rmsnorm-l1 moe-gating-l1; do
  D=$(stdpath "$t")
  if [ -f "$D/results/bench_std.json" ]; then say "[skip] $t 已有 bench_std"; continue; fi
  say "===== [$(date +%H:%M:%S)] RUN $t (dev=$DEV) ====="
  python3 -u -m harness.launch --task "$t" --arm "$ARM" --budget 1800 --max-usd 5.0 >> "$LOG" 2>&1
  if [ -f "$D/results/bench_std.json" ]; then say "[OK] $t $(date +%H:%M:%S)"
  else say "[MISS] $t $(date +%H:%M:%S)"; fi
done
say "##### ARM=$ARM DONE $(date) #####"
