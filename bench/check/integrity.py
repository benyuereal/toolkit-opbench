"""完整性：防"评测过程被改"。思路参考 SOL-ExecBench 的 reward_hack，
但检查项按我们的场景定。

为什么这些检查必须在**评分之前**跑，而不是在信任 agent 之后：
agent 有 Bash 与 Write，题面、计时函数、评测脚本都在它的可写范围内。
只要有一项被改，后面的数字全部作废——所以先判完整性，再谈分数。

每一类都做成"快照 → 比对"两段式：快照一定要在 agent 开始工作**之前**取。
"""
import hashlib
import os
import threading


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(path: str, exts: tuple[str, ...] | None = None) -> dict[str, str]:
    """{相对路径: sha256}。用于对整道题（题面 + 参考实现 + 机检）做快照。"""
    out: dict[str, str] = {}
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if exts and not fn.endswith(exts):
                continue
            p = os.path.join(root, fn)
            out[os.path.relpath(p, path)] = hash_file(p)
    return out


def verify_task_unchanged(task_dir: str, expected_hash: str) -> bool:
    """task.md 哈希必须与评测前一致——防止 agent 改题面适配自己。"""
    p = os.path.join(task_dir, "task.md")
    if not os.path.isfile(p):
        return False
    return hash_file(p) == expected_hash


def diff_hashes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """返回变化的文件（新增/删除/改动）。空表示没变。"""
    problems = []
    for name in sorted(set(before) - set(after)):
        problems.append(f"文件被删除：{name}")
    for name in sorted(set(after) - set(before)):
        problems.append(f"出现新文件：{name}")
    for name in sorted(set(before) & set(after)):
        if before[name] != after[name]:
            problems.append(f"文件被修改：{name}")
    return problems


def snapshot_functions(namespace: dict, names: list[str]) -> dict[str, int]:
    """记录关键函数 id()，跑完比对，检测被替换。

    只记存在的那几个；缺的不记（缺 = 后面 check 时报"消失"，
    那是另一种问题，不该混进"被替换"）。
    """
    return {n: id(namespace[n]) for n in names if n in namespace}


def check_functions_unchanged(snap: dict, namespace: dict) -> list[str]:
    problems = []
    for name, ident in snap.items():
        if name not in namespace:
            problems.append(f"关键函数消失：{name}")
        elif id(namespace[name]) != ident:
            problems.append(f"关键函数被替换：{name}")
    return problems


def check_no_background_threads(before: int, after: int) -> bool:
    """线程数不变才算干净。

    只比"多了没少"：agent 自己的线程池在评测结束时被回收是正常的，
    评测结束后**新出现且仍在活着的**线程才是注入。调用方传
    `threading.active_count()` 的快照即可。
    """
    return after <= before


def active_thread_names() -> list[str]:
    """审计用：当前线程名。发现异常线程时能直接看出是哪来的。"""
    return sorted(t.name for t in threading.enumerate())


def check_no_lazy_tensors(outputs) -> list[str]:
    """检测未实际计算的 lazy/proxy 张量。

    "lazy" 的判法不靠类型名猜，靠三件可测的事：
      1. 是 torch.Tensor 但 device.type 为 'meta'（形状对、数据没有）
      2. 计算后数值里出现 NaN/Inf 而题目没声明允许（调用方另判）
      3. 声称的 shape 与 tensor.shape 不符（见 check_shape_claim）
    这里只做第 1 条与"是不是 Tensor"的判定。
    """
    problems = []
    items = outputs.items() if isinstance(outputs, dict) else enumerate(outputs or [])
    for name, value in items:
        mod = type(value).__module__
        if mod.startswith("torch"):
            try:
                dev = value.device.type
            except Exception:
                continue
            if dev == "meta":
                problems.append(f"{name} 是 meta 张量（只有形状，没有数据）")
            continue
        # 非 torch 对象：检查是否带 __torch_function__ 之类的 proxy 协议
        if hasattr(value, "__torch_function__") or hasattr(value, "__torch_dispatch__"):
            problems.append(f"{name} 是未求值的 proxy 对象（{type(value).__name__}）")
    return problems


def check_shape_claim(claimed: dict, actual: dict) -> list[str]:
    """它说跑了 M=4096，实际跑的是不是 4096。防"小形状冒充大形状"。"""
    problems = []
    for key, want in (claimed or {}).items():
        if key not in (actual or {}):
            problems.append(f"声称了 {key}={want}，但结果里没有这个 shape")
            continue
        got = actual[key]
        if list(got) != list(want):
            problems.append(f"{key} 声称 {want}，实际 {got}")
    return problems


def snapshot(workdir: str) -> dict:
    """一次取齐完整性快照：题面哈希、题目目录哈希、线程数。"""
    return {
        "task_md_sha256": hash_file(os.path.join(workdir, "task.md"))
        if os.path.isfile(os.path.join(workdir, "task.md")) else None,
        "threads": threading.active_count(),
    }
