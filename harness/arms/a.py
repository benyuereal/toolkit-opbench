"""A 臂（裸）：只给 task.md，什么都不加。

隔离靠两件事：以 opeval 用户运行（读不到 chmod 700 的 skill 仓库）、
cwd 里只有题目本身。任何"顺手给点提示"都会污染基线，一律不做。
"""
ARM = "A"
USER = "opeval"


def build_prompt(task_md: str) -> str:
    return task_md


def extra_env() -> dict:
    return {}
