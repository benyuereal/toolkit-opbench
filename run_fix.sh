#!/bin/bash
cd /workspace/toolkit-opbench || exit 1
export OPBENCH_ENV=/workspace/opendesign-sk.env
LOG=runs/fix_A_$(date +%m%d_%H%M).log
echo "[fix] waiting for main batch to finish..." >> "$LOG"
while tmux has-session -t opbench 2>/dev/null; do sleep 30; done
echo "[fix] main done at $(date)" >> "$LOG"
for t in mla-sparse-l1 norm-fused-add-rmsnorm-l1 moe-gating-l1; do
  d="/tmp/opbench-run/arm-A/$t/attempt-1"
  if [ -f "$d/results/bench_std.json" ]; then
    echo "[fix][skip] $t/A 已有 bench_std" >> "$LOG"; continue
  fi
  echo "[fix][run] $t/A $(date +%H:%M:%S)" >> "$LOG"
  rm -rf /tmp/opbench-run/arm-A/$t/attempt-*
  python3 -u -m harness.run --task "$t" --arm A --budget 1800 --max-usd 5.0 >> "$LOG" 2>&1
  if [ -f "$d/results/bench_std.json" ]; then
    echo "[fix][OK] $t/A $(date +%H:%M:%S)" >> "$LOG"
  else
    echo "[fix][MISS] $t/A $(date +%H:%M:%S)" >> "$LOG"
  fi
done
echo "[fix] ALL DONE $(date)" >> "$LOG"
