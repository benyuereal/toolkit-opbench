#!/bin/bash
cd /workspace/toolkit-opbench || exit 1
export OPBENCH_ENV=/workspace/opendesign-sk.env
MAXUSD=5.0
LOG=runs/batch_std_$(date +%m%d_%H%M).log
say(){ echo "$*" | tee -a "$LOG"; }
has_std(){
  t=$1; a=$2
  if [ "$a" = "A" ]; then d="/tmp/opbench-run/arm-A/$t/attempt-1"; else d="runs/arm-$a/$t/attempt-1"; fi
  [ -f "$d/results/bench_std.json" ]
}
run(){
  t=$1; a=$2; b=$3
  if has_std "$t" "$a"; then say "[skip] $t/$a (bench_std 已有)"; return; fi
  say "===== [$(date +%H:%M:%S)] RUN $t/$a (${b}s) ====="
  python3 -u -m harness.run --task "$t" --arm "$a" --budget "$b" --max-usd $MAXUSD >> "$LOG" 2>&1
  rc=$?
  if has_std "$t" "$a"; then say "[rc=$rc][OK bench_std] $t/$a $(date +%H:%M:%S)"
  else say "[rc=$rc][MISS bench_std] $t/$a $(date +%H:%M:%S)"; fi
}
for t in mla-sparse-l1 norm-fused-add-rmsnorm-l1 moe-gating-l1; do
  for a in A C D; do run "$t" "$a" 1800; done
done
say ""
say "##### STD ROUND DONE $(date) #####"
