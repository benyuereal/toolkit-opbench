"""B 臂（库清单）：只多给 libraries-registry.yaml。

测的是"知道有哪些库"值多少，所以**只给这一个文件**，
不给 atlas 卡、不给 scenarios、不给 failure-modes。
"""
ARM = "B"
LIB_REGISTRY = (
    "/workspace/toolkit_skills/data/ops/libraries-registry.yaml"
)


def build_prompt(task_md: str) -> str:
    raise NotImplementedError


def stage_files(cwd: str) -> list[str]:
    """把 registry 拷进 cwd（只读），返回拷贝清单。"""
    raise NotImplementedError
