# 工作流：一次评测的六个步骤

```bash
python3 -m harness.doctor                          # 1 体检
python3 -m harness.run --suite suites/l5-only.yaml --prepare   # 2 准备
python3 -m harness.run --suite suites/l5-only.yaml             # 3 执行
python3 -m bench.check --suite suites/l5-only.yaml             # 4 校验
python3 -m bench.score --suite suites/l5-only.yaml             # 5 打分
python3 -m harness.collect --suite suites/l5-only.yaml         # 6 报告
```

## 1. doctor：能不能测

| 检查 | 红项怎么办 |
|---|---|
| 8 张卡可见 | 容器没起对，重启 |
| `torch.cuda.Event` 可计时 | 环境问题，别继续 |
| L2 = 8 MB | 与 constants 不符 → 更新 constants，不是改代码 |
| 当前 sclk | **600 MHz 说明在降频**，必须 ramp 后再测 |
| 能否锁频 | 不行就记录档位，报告里注明"未锁频" |
| 库版本 | 缺版本 = 结果不可复跑 |
| 另一会话占卡 | 等，或换卡 |

## 2. prepare：这次跑什么

按 suite 展开成 `(task, arm)` 队列，建好 `runs/<arm>/<task>/`（chmod 777，A 臂要以 opbench 写）。

## 3. run：执行契约

1. 注入上下文（A 无 / B 只给库清单 / C 给 skill 仓库）
2. **预热到工作档 sclk**，记录
3. `timeout <budget> claude -p "$(cat task.md)"`
4. 收集 `report.md` / `scripts/` / `results/`
5. 写 `trace.jsonl`：wall time、tokens、sclk、exit code

**超时即失败，不续跑**——补跑破坏固定预算的可比性。

## 4. verify：只认产物

| 检查 | 模块 |
|---|---|
| 数值正确（含 holdout + 压力数值） | `bench/check/correctness.py` |
| 题面未改 / 无劫持 / 无 lazy | `bench/check/integrity.py` |
| A 臂没偷看 | `bench/check/leak.py` |
| 声明的函数真的存在 | `bench/check/hallucination.py` |

任一不过，该单元记 0，**但轨迹保留**——失败样本比成功样本信息量大。

## 5. score：算分

`ratio` / `sol_like` / `judgment`，以及跨形状几何平均。缺 `baselines/<id>.json` 时先跑一次基线并存档（这一步慢，单独跑）。

## 6. report：出结论

三臂对比表 + 三个 Δ + 泄漏与污染声明。格式见 `docs/report-template`（待补）。

---

## 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 测出的带宽比峰值还高 | 没 flush L2 | `bench/timing/flush.py`，buffer ≥16MB |
| 同一 kernel 两次差一倍 | 时钟档位不同 | 记 sclk，ramp 后测 |
| 重叠度两次结果相反 | 两侧耗时量级不匹配 | 量级比须 ∈[0.2,5]，否则拒出结论 |
| A 臂表现接近 C 臂 | 可能偷看了 | 查泄漏指纹表 |
