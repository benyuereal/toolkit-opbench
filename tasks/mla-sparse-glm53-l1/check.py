"""题专属机检（可选）。返回问题列表，空表示通过。

本题输出是连续 bf16，通用比较器（`|a-e| <= atol + rtol*|e|`）就够用：
没有"顺序不唯一"或"只有前缀有效"的约定。所以不需要 `compare_task`。

容差含义（rtol=0.05 / atol=0.005）：
* 实测库实现与 fp32 参考的 max_abs 约 2e-3、mean_abs 约 5e-5，
  在该容差下 0 个元素超差（S=128 与 S=2048 都测过）。
* 把哨兵 `-1` 当索引 0 去 gather 时，S=128 有 204 万/210 万元素超差、
  max_abs 2.9 —— 这个错误**不会**被容差放过。
"""


def check(workdir: str, results: dict) -> list[str]:
    return []
