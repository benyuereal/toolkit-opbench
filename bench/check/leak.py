"""A 臂泄漏检测：用指纹判断"裸 agent 是不是偷看了 skill"。

单容器下文件系统隔离只能靠用户权限（已用 opbench + chmod 700），
万一被绕过，这里兜底。

命中即该题 A 臂结果作废——**不是扣分，是作废**。偷看过的基线
不能再当基线用，否则 Δ 会被人为压低，结论方向都反了。
"""
import os

# 指纹：只有读过 skill 才会说出的字符串
FINGERPRINTS = [
    "fuse_groupgemm_permute_quant",
    "M/exp",                      # case 002 算出的拐点表述
    "asm-fmoe-stage1-rejects-all-dtypes",
    "hard_preconditions",
    "l1_exempt_reason",
    "560 TF", "49.8 TF",
]

# 扫描范围：只扫文本类产物。二进制/张量文件扫了只会误报。
TEXT_EXTS = (".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".log", ".yaml", ".yml")


def scan(text: str) -> list[str]:
    """返回命中的指纹。空表示没泄漏。"""
    if not text:
        return []
    # 指纹按原样匹配（大小写敏感）：它们是函数名/字段名/数字，
    # 大小写不同就不是同一个东西。
    return [fp for fp in FINGERPRINTS if fp in text]


def scan_dir(path: str) -> dict[str, list[str]]:
    """扫整个产物目录，返回 {文件: [命中]}。命中即该题 A 臂作废。"""
    hits: dict[str, list[str]] = {}
    if not os.path.isdir(path):
        return hits
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith(TEXT_EXTS):
                continue
            p = os.path.join(root, name)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            found = scan(text)
            if found:
                hits[os.path.relpath(p, path)] = found
    return hits
