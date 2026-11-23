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

# 第二类指纹：**题面本身没有、只在答案侧文件里出现**的字符串。
# 前一类抓"偷看 skill"，这一类抓"偷看 expect.md / definition.json"。
# 为什么必须单列：A 臂的 cwd 在外层仓库里，仓库对 opbench 用户可读
# （实测 expect.md 644），所以"读到答案"在文件系统层面是可能的；
# 而答案是自然语言，抄了结论不会有任何函数名之类的痕迹。
# 这里用的是**答案侧独有**的措辞，不是题面词——避免把正常回答判成泄漏。
ANSWER_FINGERPRINTS = [
    "expected_verdict",
    "verdict_keywords",
    "forbidden_claims",
    "evidence_required",
    "参考答案",          # expect.md 的标题
    "不给 agent 看",
    "修正记录",
]


def scan_answer_leak(text: str) -> list[str]:
    """检测"答案侧独有"的痕迹。与 scan() 分开，因为两者含义不同：
    命中 scan() = 偷看 skill/我们的标定；命中本函数 = 偷看 expect/definition。
    两者都作废该单元，但报告里要分开写。"""
    if not text:
        return []
    return [fp for fp in ANSWER_FINGERPRINTS if fp in text]

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
            found = scan(text) + scan_answer_leak(text)
            if found:
                hits[os.path.relpath(p, path)] = found
    return hits
