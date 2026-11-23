"""A 臂泄漏检测：用指纹判断"裸 agent 是不是偷看了 skill"。

单容器下文件系统隔离只能靠用户权限（已用 opbench + chmod 700），
万一被绕过，这里兜底。
"""

# 指纹：只有读过 skill 才会说出的字符串
FINGERPRINTS = [
    "fuse_groupgemm_permute_quant",
    "M/exp",                      # case 002 算出的拐点表述
    "asm-fmoe-stage1-rejects-all-dtypes",
    "hard_preconditions",
    "l1_exempt_reason",
    "560 TF", "49.8 TF",
]


def scan(text: str) -> list[str]:
    """返回命中的指纹。空表示没泄漏。"""
    raise NotImplementedError


def scan_dir(path: str) -> dict[str, list[str]]:
    """扫整个产物目录，返回 {文件: [命中]}。命中即该题 A 臂作废。"""
    raise NotImplementedError
