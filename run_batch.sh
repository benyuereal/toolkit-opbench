#!/bin/bash
cd /workspace/toolkit-opbench || exit 1
export OPBENCH_ENV=/workspace/opendesign-sk.env
MAXUSD=5.0
LOG=runs/batch_$(date +%m%d_%H%M).log
say(){ echo "$*" | tee -a "$LOG"; }
has_attempt(){
  t=$1; a=$2
  if [ "$a" = "A" ]; then d="/tmp/opbench-run/arm-A/$t/attempt-1"; else d="runs/arm-$a/$t/attempt-1"; fi
  [ -f "$d/meta.json" ]
}
run(){
  t=$1; a=$2; b=$3
  if has_attempt "$t" "$a"; then say "[skip] $t/$a"; return; fi
  say "===== [$(date +%H:%M:%S)] RUN $t/$a (${b}s) ====="
  python3 -u -m harness.run --task "$t" --arm "$a" --budget "$b" --max-usd $MAXUSD >> "$LOG" 2>&1
  say "[rc=$?] $t/$a $(date +%H:%M:%S)"
}
# L1：只缺 moe-gating-l1/C，先跑，齐了就能出完整 L1 对比
run moe-gating-l1 C 1800
# L5 六题三臂
for t in gemm-fp8-l5 moe-asm-fmoe-l5 moe-shared-expert-l5 attn-gqa-repeat-kv-l5 lin-hadamard-l5 pipe-allreduce-l5; do
  for a in A C D; do run "$t" "$a" 600; done
done
say ""
say "##### ALL DONE $(date) #####"
