"""D 臂（SkillHub）：给 /workspace/SkillHub，只读。

与 C 臂（toolkit_skills）的区别：
  * C 臂是**我们自己的**算子知识库（atlas 卡 / scenarios / failure-modes）
  * D 臂是**团队共享的** Agent Skills 目录（44 个 skill，按 catalog 分组）

两者是不同来源的知识层，所以做对照实验时是两条独立的臂。

## 为什么单开一个臂而不是复用 C

如果让 D 臂也软链到 toolkit_skills，那测的就是同一份东西。
必须指向**不同的根目录**，且把各自的 `git rev-parse HEAD` 记进 trace——
否则"这一轮看的是哪个版本的知识库"无从追溯。
"""
import os

ARM = "D"
SKILLHUB_ROOT = "/workspace/SkillHub"
LINK_NAME = "skillhub"


def build_prompt(task_md: str) -> str:
    """在题面后追加一句"可用资料在 skillhub/"，其余不改。

    措辞与 C 臂保持**结构一致**：都只说"参考资料在哪"，不说"里面有什么"。
    三臂之间只差资源声明的具体路径，这是实验有效性的前提。
    """
    return (
        f"{task_md.rstrip()}\n\n"
        f"---\n\n"
        f"参考资料：`{LINK_NAME}/`（指向团队共享的 Agent Skills 目录，只读）。\n"
    )


def stage_files(cwd: str) -> list[str]:
    """不拷文件，走软链。空清单是结果，不是遗漏。"""
    return []


def mount(cwd: str) -> list[str]:
    """软链 SkillHub 到 cwd。返回挂载清单。"""
    if not os.path.isdir(SKILLHUB_ROOT):
        raise FileNotFoundError(
            f"SkillHub 不存在：{SKILLHUB_ROOT}。缺它 D 臂就退化成了 A 臂，"
            "这轮实验作废而不是照跑"
        )
    if not os.access(SKILLHUB_ROOT, os.R_OK | os.X_OK):
        raise PermissionError(
            f"当前用户读不到 {SKILLHUB_ROOT}。D 臂必须能读——"
            "读不到的 D 臂就是 A 臂，整轮作废"
        )
    link = os.path.join(cwd, LINK_NAME)
    if os.path.islink(link) or os.path.exists(link):
        os.unlink(link)
    os.symlink(SKILLHUB_ROOT, link)
    return [LINK_NAME]


def skill_version() -> dict:
    """记录 SkillHub 的版本。"""
    import subprocess

    info = {"root": SKILLHUB_ROOT, "git_head": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", SKILLHUB_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode == 0:
            info["git_head"] = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", SKILLHUB_ROOT, "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
        if dirty.returncode == 0:
            info["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info
