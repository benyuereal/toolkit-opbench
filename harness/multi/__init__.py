"""多臂对照实验（N 臂，不只三臂），模块化拆分。

    建单元   model.py        —— 口径 + Unit 展开
    隔离     isolation.py    —— 一臂一用户，互不可见
    预检     preflight.py    —— 开跑前的全部准入检查
    下发     dispatch.py     —— 命令构造、预算一致性、执行
    进度     progress.py     —— 以产物为准的进度/卡住检测
    收集     collect.py      —— 产物归拢成可比的表
    报告     report.py       —— 性能/正确性/证据三维打分

## 为什么拆开

每一块都有自己的坑。混在一个 600 行的 run.py 里时，改一处不知道会不会
碰到另一处的约束（比如"预算一致性"和"隔离"完全无关，但都在同一个函数里
检查）。拆开后每块可以单独测、单独看。

## 用法

    python3 -m harness.multi --suite three-way --check-only
    python3 -m harness.multi --suite three-way
    python3 -m harness.multi --tasks norm-fused-add-rmsnorm-l1 --arms A,C,D
    python3 -m harness.multi --progress          # 只看进度
    python3 -m harness.multi --collect           # 只收集产物

## 臂不是写死的三个

`--arms A,C,D,E` 任意组合。新臂只需要在 `harness/arms/` 里有一个模块，
定义 `ARM`、`USER`、`build_prompt`、`stage_files`，并建好对应的 Unix 用户。
`preflight` 会检查这些。
"""
