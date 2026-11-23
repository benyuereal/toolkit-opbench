"""B 臂（库清单）：只多给 libraries-registry.yaml。

测的是"知道有哪些库"值多少，所以**只给这一个文件**，
不给 atlas 卡、不给 scenarios、不给 failure-modes。

实现要点：registry 是**拷贝**进 cwd 的（名字保持 libraries-registry.yaml），
不是软链。软链会让 agent 顺着路径摸到 /workspace/toolkit_skills，B 臂
就变成了 C 臂——这条臂的全部意义是"只给一份清单"。
"""
import os
import shutil

ARM = "B"
LIB_REGISTRY = (
    "/workspace/toolkit_skills/data/ops/libraries-registry.yaml"
)
STAGED_NAME = "libraries-registry.yaml"


def build_prompt(task_md: str) -> str:
    return (
        f"{task_md.rstrip()}\n\n"
        f"---\n\n"
        f"参考资料：当前目录下的 `{STAGED_NAME}`（本机已装算子库的清单、版本与导入名）。\n"
        f"除此之外没有其他参考资料。\n"
    )


def stage_files(cwd: str) -> list[str]:
    """把 registry 拷进 cwd（只读），返回拷贝清单。"""
    if not os.path.isfile(LIB_REGISTRY):
        raise FileNotFoundError(
            f"B 臂的库清单不存在：{LIB_REGISTRY}。"
            "缺它 B 臂就没有可追加的上下文，不能退化成一个更短的 A 臂去跑"
        )
    dst = os.path.join(cwd, STAGED_NAME)
    shutil.copyfile(LIB_REGISTRY, dst)
    os.chmod(dst, 0o444)
    return [STAGED_NAME]
