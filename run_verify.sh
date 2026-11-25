#!/bin/bash
cd /workspace/toolkit-opbench || exit 1
export OPBENCH_ENV=/workspace/opendesign-sk.env
echo "[start] $(date)"
python3 -u -m harness.run --task gemm-fp8-l5 --arm A --max-usd 3.0
echo "[exit=$?] $(date)"
