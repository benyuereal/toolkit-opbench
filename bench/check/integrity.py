"""完整性：防"评测过程被改"。针对的是同一类作弊手法：
但检查项按我们的场景定。
"""
import hashlib


def hash_file(path: str) -> str:
    raise NotImplementedError


def verify_task_unchanged(task_dir: str, expected_hash: str) -> bool:
    """task.md 哈希必须与评测前一致——防止 agent 改题面适配自己。"""
    raise NotImplementedError


def snapshot_functions(namespace: dict, names: list[str]) -> dict[str, int]:
    """记录关键函数 id()，跑完比对，检测被替换。"""
    raise NotImplementedError


def check_functions_unchanged(snap: dict, namespace: dict) -> list[str]:
    raise NotImplementedError


def check_no_background_threads(before: int, after: int) -> bool:
    raise NotImplementedError


def check_no_lazy_tensors(outputs) -> list[str]:
    """检测未实际计算的 lazy/proxy 张量。"""
    raise NotImplementedError


def check_shape_claim(claimed: dict, actual: dict) -> list[str]:
    """它说跑了 M=4096，实际跑的是不是 4096。防"小形状冒充大形状"。"""
    raise NotImplementedError
