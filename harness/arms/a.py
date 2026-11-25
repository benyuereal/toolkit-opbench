"""A 臂（裸）：只给 task.md，什么都不加。

隔离靠两件事：以 opbench 用户运行（读不到 chmod 700 的 skill 仓库）、
cwd 里只有题目本身。任何"顺手给点提示"都会污染基线，一律不做。

三臂只差"一行资源声明"（README 第四节）。所以 A 臂也要显式写出
`参考资料：无`，而不是干脆不写——否则 A 的 prompt 比 B/C 短一截，
测出来的是提示长度而不是知识。这一行不是提示，是**实验条件声明**。
"""
ARM = "A"
USER = "opb_a"


def build_prompt(task_md: str) -> str:
    # 与 C/D 逐字相同：差异只能来自"有没有 skill 环境"，不能来自 prompt。
    return task_md


def extra_env() -> dict:
    return {}


def stage_files(cwd: str) -> list[str]:
    """A 臂不注入任何文件。空清单是结果，不是遗漏。"""
    return []
