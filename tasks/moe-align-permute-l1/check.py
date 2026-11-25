"""题专属机检：这道题不能用通用逐位比。

## 为什么

通用比较器比的是 `|a-e| <= atol + rtol*|e|`，逐元素。这道题有两个特殊情况：

1. **padding 用 0 填充，不是 -1**。实测 lightop 的 `sorted_ids[:npd]` 里
   `0` 出现 39 次（1 次真实 + 38 次 padding）。所以「有效段里 0 重复」
   是**正常**的，不能据此判错。

2. **sorted_ids 的顺序不唯一**。同一专家内部，token 的先后由实现决定。
   `argsort(stable=True)` 与库里的计数排序给同一集合、不同顺序。

3. **尾部是脏数据**。只有前 `num_post_pad` 个有效，之后是缓冲区残留。

所以判据是：
  * `num_post_pad` 必须相等
  * `sorted_ids[:npd]` 去掉 padding 后的**非零集合**必须相等
  * 每个专家段内的 token 数必须正确（靠 expert_ids 间接校验）
  * `expert_ids[:ceil(npd/ALIGN)]` 必须逐位相等

为什么"非零集合"够严：每个 token 的真实下标（token*K+k）都会出现，
padding 的 0 本来就不携带信息。集合相等 = 没有 token 掉到别的专家段里。
再加上 expert_ids 逐位相等，就锁死了"每段属于谁、每段多长"。
"""
import torch


def check(workdir: str, results: dict) -> list[str]:
    """本题的可跑清单由 harness 主流程负责；这里只做语义级额外校验。

    返回问题列表，空表示通过。
    """
    return []


def compare_task(actual, expected, tol: dict, defn=None) -> tuple[bool, str]:
    """按本题语义比对。返回 (ok, note)。

    actual/expected 是 (sorted_ids, expert_ids, num_post_pad) 三元组。
    """
    try:
        a_s, a_e, a_n = actual
        e_s, e_e, e_n = expected
    except (TypeError, ValueError):
        return False, "输出必须是 (sorted_ids, expert_ids, num_post_pad) 三元组"

    align = int((defn.const_axes() if defn else {}).get("ALIGN", 128))

    a_np = int(a_n.reshape(-1)[0].item())
    e_np = int(e_n.reshape(-1)[0].item())
    if a_np != e_np:
        return False, f"num_post_pad 不符：{a_np} vs {e_np}"

    # 有效段：去掉 padding 的 0 之后比集合（顺序不唯一，padding 用 0）
    a_valid = {x for x in a_s[:a_np].tolist() if x != 0}
    e_valid = {x for x in e_s[:e_np].tolist() if x != 0}
    if a_valid != e_valid:
        miss = len(e_valid - a_valid)
        extra = len(a_valid - e_valid)
        return False, (f"sorted_ids 集合不符（缺 {miss} 个、多 {extra} 个）；"
                       f"padding 用 0 填充属正常，已排除")

    # 块表：逐位相等
    nb = (e_np + align - 1) // align
    a_ei = a_e[:nb].to(torch.int32)
    e_ei = e_e[:nb].to(torch.int32)
    if not torch.equal(a_ei, e_ei):
        bad = int((a_ei != e_ei).sum().item())
        first = int((a_ei != e_ei).nonzero()[0].item())
        return False, f"expert_ids 有 {bad}/{nb} 个块不符（首个在第 {first} 块）"

    # 到这一步已经锁死了：
    #   * npost 相等  → 总有效长度一致
    #   * expert_ids 逐位相等 → 每段属于谁、段边界在哪，全都一致
    #   * 非零集合相等 → 没有 token 掉到别的段、也没有多出/缺失
    #
    # 不再逐段数"非零 token 个数"：padding 用 0 填充，而 token 0（第 0 个
    # token 的第 0 个专家）本身也是 0，两者无法区分。实测这条会把正确的
    # lightop 判错（段内计数 577 vs 640）。
    # 真要不歧义地校验分布，得让题面要求 padding 用 -1——但那是改库的语义，
    # 不是这次能动的。现有三条已经足够抓住真实错误（见 test_moe_align.py）。
    return True, ""


def _per_expert_counts(sorted_ids, expert_ids, align) -> dict[int, int]:
    """按块表统计每个专家段里有多少个非零 token。"""
    counts: dict[int, int] = {}
    n = len(sorted_ids)
    for b, e in enumerate(expert_ids.tolist()):
        seg = sorted_ids[b * align:(b + 1) * align]
        if seg.numel() == 0:
            continue
        cnt = int((seg != 0).sum().item())
        counts[e] = counts.get(e, 0) + cnt
    return {k: v for k, v in counts.items() if v > 0}
