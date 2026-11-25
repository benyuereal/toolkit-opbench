本目录是三臂对比实验（norm-fused-add-rmsnorm-l1）的原始数据。

⚠️ arm-*_bench_std.json 是 2026-11-25 09:14 的**中间快照**。
   三臂当时仍在运行（budget 3600s，08:47 启动，09:47 结束），终值会变。
   判断是否终值：看 attempt 目录里有没有 meta.json（harness 只在结束时写）。

arm-{A,C,D}_bench_std.json  性能数据（唯一被 harness 认的格式）
arm-C_loop.jsonl            C 臂 loop.py 的迭代流水（如有）

结论摘要见上级目录 HANDOFF.md 第 3 节（已标注为中间结果）。

生成时间：2026-11-25 09:14
