"""C 臂（完整 skill）：给 /workspace/toolkit_skills，只读。

这是被测对象的真实形态。注意：被测对象是它，所以**改它要另开 commit**，
否则测的是一个一直在变的靶子。
"""
ARM = "C"
SKILL_ROOT = "/workspace/toolkit_skills"


def build_prompt(task_md: str) -> str:
    """在题面后追加一句"可用资料在 SKILL_ROOT"，其余不改。"""
    raise NotImplementedError


def mount(cwd: str) -> None:
    """软链或只读挂载。不允许写。"""
    raise NotImplementedError
