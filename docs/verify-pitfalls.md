# 复测阶段的坑（2026-11-25 实录）

这份文档记录**验证三臂提交物**时踩到的问题。目的不是追责，是下次出题、
下发、复测时不再重复。

## 一、根本原因：没有提交契约

三臂都按要求交了 `bench_std.json`，但 `impl` 字段没有格式约束，于是：

| 臂 | 实际写的 impl |
|---|---|
| A | `triton 3.6.0 :: scripts/fused_add_rmsnorm_swiglu.py::fused_add_rmsnorm_swiglu` |
| C | `fused-hip-v5 = 自研 HIP kernel kernels/norm_act_swiglu_v5.hip (hipcc, ...)` |
| D | `triton.jit:_fused_kernel (custom fused Triton kernel, ...)` |

三种写法，只有一个（A）能直接定位到可执行代码。C 写的是**变体标签**，
真身藏在算子卡的 `_common.py:get_runner("fused-hip-v5")`。D 写的
`_fused_kernel` 在它的任何文件里都**不存在**（真身叫 `forward`，在
`fused_final.py`），而它的 scripts/ 下有 18 个 .py，其中至少两个跑不通。

**代价**：复测方只能靠猜，猜错了会把 A 的成绩记成 D 的。

### 已做的修复

`harness/measure_spec.py:_impl_contract()` 把格式写死了，并且
**题面注入时就下发**（每个 agent 都会看到）：

```
<脚本相对路径>:<入口函数名>  —  <一句话说明>
```

同时加了 6 条硬性要求（见该函数）。最关键的几条：

- 签名必须是 `fn(<inputs 键序>, eps=<默认值>)`，**标量不许插在张量中间**
- 内核函数名不要用 `_fused_kernel` 这类通用名
- impl 解析不出来 → 本轮判"无法复现"，不参与排名

## 二、五个具体的坑

### 坑 1：并行跑导致 HBM 争用

三臂在卡 4/5/6 上并行，但**HBM 是整机共享的**。同一个 kernel：

| | 自报 | 干净卡复测 | 差 |
|---|---|---|---|
| A | 0.17328 | 0.17536 | +1% |
| D | **0.11504** | **0.17440** | **+52%** |

D 报了 5 次重复、波动 ±0.1%，看起来极可信，但换张干净卡就复现不出来。
A 和 C 都在报告里主动指出了这件事。

**结论**：并行跑能测"谁写出了什么实现"，不能测"谁的实现更快"。
排名一律用复测。

### 坑 2：Triton 内核同名，JIT 缓存串味

A 和 D 的内核都叫 `_fused_kernel`，但签名不同：

```
A: _fused_kernel(X, RES, W, OUT, M, eps, H, I, ...)
D: _fused_kernel(X, R, W, OUT, eps, H, I)
```

同一进程里先加载 A 再调 D，Triton 按 (函数名, 参数签名) 命中缓存，
把 D 的调用路由到了 A 的编译产物，报：

```
Unsupported ptr type <['2048'], fp32> in `tl.store`
```

那个报错和 D 的代码毫无关系，为此查了四轮。

**已修**：`harness/verify.py:_run_isolated()` 改成**每个实现一个子进程**。

### 坑 3：extra_args 按位置传，撞上 out 参数

D 的签名：

```python
def forward(x, residual, weight, out=None, eps=1e-6, ...)
```

第 4 个位置是 **`out`**，不是 `eps`。复测按位置传 `[1e-6]` 就变成
`out=1e-6` → triton 把 float 当指针 → 同一个 `Unsupported ptr type` 报错。

**已修**：`extra_args` 改为**按关键字**补（试 `eps` / `epsilon` /
`variance_epsilon`），对不上才退回位置传。

### 坑 4：一次 attempt 里有几十个中间版本

D 的 scripts/ 下 18 个 .py：`tune.py` / `tune2.py` / ... / `tune5.py` /
`fused_triton.py` / `fused_final.py` / `bench.py` / `variance.py` / ...

自动识别按"文件名带 fused/kernel"挑，挑中了 `fused_triton.py`——
那是个**编译不过的旧版本**。

**已修**：`harness/verify.py:pick_submission()` 只认 `bench_std.json`
的 `impl` 字段，并从 report 里交叉验证"报告声称的版本"与 impl 是否一致。

### 坑 5：`--max-usd` 把长任务腰斩

早期默认 `--max-usd 5.0`，实测在 ~33 分钟切断 60 分钟的题，三臂被切在
近似但不同的时刻，看起来像"完成了"。

**已修**：`DEFAULT_MAX_USD = None`，预算只留 wall time。

## 三、复测流程（固化后）

```bash
# 1. 看每条臂最终提交的是哪一版
python3 -m harness.verify --task <id> --arms A C D

# 2. 认不出时手工指定
python3 -m harness.verify --task <id> \
  --impl "A=/path/to/k.py:fused" --impl "D=/path/to/k.py:forward"

# 3. 只做版本判定，不测
python3 -c "from harness.verify import pick_all, render_pick; \
  print(render_pick(pick_all('<id>', ['A','C','D'])))"
```

复测一律：
- **逐个子进程**（避免 JIT 缓存串味）
- **多轮**（`--rounds`，默认 3）取跨轮中位数
- **绑一张干净卡**（`--device`）
- 每轮前 flush L2

## 四、还没解决的

1. **C 臂的实现要靠 `get_runner` 解析**——它交的是 HIP `.so`，Python 入口
   是卡里的工厂函数。这意味着复测依赖被测对象的内部结构。更干净的做法
   是要求在 impl 里指向一个自带薄封装的 `.py`。
2. **`sclk` 不同卡不同档**（卡 4=1350，卡 5/6=1200，其余 600）。复测时
   必须同卡，否则差异里混了频率。
3. **压力输入（large ×1e4）会让绝对容差失效**——bf16 在量级 1e5 下
   合法地产生 max_abs=1536。判据要用尺度无关的量（rel_l2、max_abs/max|e|）。

---

## 五、计时口径：CPU 开销不能算成 kernel 时间

### 问题

原口径：

```python
e1.record()
call()          # ← 这里的 Python 时间会被算进 event 区间
e2.record()
```

单次调用时 GPU 队列是空的。若 `call()` 的 CPU 侧耗时超过 GPU 执行时间，
GPU 在 CPU 提交期间**空闲等待**，那段时间全被记进 event 区间。

实测（C 臂，M=8192）：

| 口径 | 结果 |
|---|---|
| `record → call → record` | **0.221 ms** |
| 队列预热后 | **0.131 ms** |
| 批量 20 次摊销 | 0.136 ms |

差 62%。而这不是选手慢——它的 kernel 本体确实只有 131µs。

### 根因

C 的包装层有个字符级 bug：

```python
if "lib" not in _LIB or so not in _LIB:   # "lib" 永远不在 _LIB，条件恒真
```

`_LIB` 的 key 是 `"v5r2.so"`，代码却查 `"lib"` → 每次调用都重新
`ctypes.CDLL` 加载 `.so`（`pathlib.stat` + `getattr` + `__build_class__`），
CPU 侧多 60~77µs。

### 修法：队列预热

```python
for _ in range(PRIME):   # 3 次，不计时
    call()
flush()                  # 清 L2
e1.record()
call()                   # GPU 还在消化前几次，CPU 早已提交完
e2.record()
```

event 区间里只剩第 4 次的 GPU 时间。**保留单次 launch 的真实代价**，
只是不把 Python 包装层开销算进去。

### 与 CUDA Graph 的区别

| | CUDA Graph | 队列预热 |
|---|---|---|
| 消除 | CPU 提交 + **launch 开销** | 只消除 CPU 提交 |
| launch 代价 | 被藏掉 | 保留 |
| 适用 | decode 阶段小算子（vLLM/SGLang 确实这么跑） | 通用、带宽题 |

对带宽主导的题（swiglu 167MB），两种口径差别不大。
对 launch 主导的小算子（moe-gating 2.5MB），用 Graph 会把要考的东西藏掉——
那类题的 launch 地板（43µs）是真实成本，不能抹掉。

### 同时保留 CPU 开销的观测

复测会另外量一次 CPU 侧耗时并报出来（`cpu_ms` / `cpu_share`）。
超过 GPU 时间 20% 会点名——那说明交付入口本身有固定开销，
**在真实推理里是真的会拖慢服务的**，只是不该算进"kernel 多快"。

---

## 六、三个测量异常及其根因（2026-11-25 实测）

### 异常 1：M=128 时"墙钟 0.056 < 纯 GPU 0.096"

物理上不可能。根因是**量了异步提交时间**：

```python
t0 = time.perf_counter()
call()                    # 只把 kernel 塞进队列，GPU 还在跑
t1 = time.perf_counter()  # ← 0.056ms
```

补上 `synchronize()` 后是 0.090ms，与 event 计时一致。

**规则**：墙钟必须配 `synchronize()`，否则量的是 CPU 提交，不是执行。

### 异常 2：M=1024 比 M=128 还快

| M | 数据量 | A 耗时 |
|---|---|---|
| 128 | 2.6 MB | 0.108 |
| 1024 | 21 MB | **0.098** |
| 8192 | 168 MB | 0.118 |

数据少一个数量级反而更慢。因为 M=128 时只有 128 行、GPU 占用率极低，
虽然数据能塞进 L2（8MB），但并行度不够；M=1024 时占用率上来了。

**影响**：这题的"最优实现"随 M 变化——M=128 时 C 赢，M≥1024 时 A 赢。
所以 M 档位选多大的，直接决定排名。

### 异常 3：首样本比稳态高 89%

```
A: [0.21696, 0.11616, 0.11456, 0.11552, ...]
    ↑ 首个样本 0.217，之后全 0.115
```

warmup=5 时更明显：`[0.24448, 0.14832, 0.11536, 0.2648, ...]`
——第 1、2、4 次都有尖峰。

根因是 Triton 的 `_LAUNCH_CACHE` 冷启动（A 的代码里手工缓存的
`CompiledKernel` 前几次未命中）。D 和 C 没有这个现象。

**规则**：warmup 必须 ≥ 30。warmup=20 时侥幸没事，warmup=5 就会采到尖峰。

### 对应的口径修订

| 项 | 原 | 现 |
|---|---|---|
| warmup | 20 | **30** |
| iters（每轮） | 50 | **10** |
| rounds | 3 | **5–6** |
| 顺序 | 固定 A→C→D | **正序/倒序交替** |
| 墙钟 | 无 sync | 必须带 sync |

正倒交替实测效果：各臂离散度从 ~2% 降到 **0.5–0.8%**。
