"""C 臂（完整 skill）：给 /workspace/toolkit_skills，只读。

这是被测对象的真实形态。注意：被测对象是它，所以**改它要另开 commit**，
否则测的是一个一直在变的靶子。

实现要点：
  * 用软链把 skill 根挂进 cwd（`skill/`）。直接 `--add-dir` 指向原路径
    会让 cwd 里没有痕迹，产物与命令都无法从目录本身复现。
  * **不拷贝**：60 MB、含 .git；拷贝既慢，又让"被测对象是哪个版本"
    变得模糊。软链 + 记录 `git rev-parse HEAD` 才是可复跑的。
  * 只读由源目录的权限保证（700 root），这里再校验一次：C 臂必须读得到，
    读不到就是环境没起对，不能静默降级。
"""
import os

ARM = "C"
SKILL_ROOT = "/workspace/toolkit_skills"
LINK_NAME = "skill"


def build_prompt(task_md: str) -> str:
    """在题面后追加一句"可用资料在 SKILL_ROOT"，其余不改。"""
    return (
        f"{task_md.rstrip()}\n\n"
        f"---\n\n"
        f"参考资料：`{LINK_NAME}/`（指向本机算子优化知识库，只读）。\n"
    )


def stage_files(cwd: str) -> list[str]:
    """C 臂不拷文件，走软链（见 mount）。空清单是结果，不是遗漏。"""
    return []


def mount(cwd: str) -> list[str]:
    """软链或只读挂载。不允许写。返回挂载清单。

    返回清单而不是 None：执行器要把挂进去的东西记进 trace，
    否则"这一轮 C 臂看到的是哪个 skill 版本"就无从追溯。
    """
    if not os.path.isdir(SKILL_ROOT):
        raise FileNotFoundError(
            f"C 臂的知识库不存在：{SKILL_ROOT}。"
            "它不存在时 C 臂就退化成了 B 臂，这轮实验作废而不是照跑"
        )
    if not os.access(SKILL_ROOT, os.R_OK | os.X_OK):
        raise PermissionError(
            f"当前用户读不到 {SKILL_ROOT}。C 臂必须能读——"
            "protocol.md 2.1 只把 **A 臂**换成 opbench 用户；B/C 臂必须用能读 skill 的"
            "用户跑（通常是 root）。读不到 skill 的 C 臂就是 B 臂，整轮作废"
        )
    link = os.path.join(cwd, LINK_NAME)
    if os.path.islink(link) or os.path.exists(link):
        os.unlink(link)
    os.symlink(SKILL_ROOT, link)
    return [LINK_NAME]


def skill_version() -> dict:
    """记录被测 skill 的版本。C 臂的结果离开这个就没法解释。"""
    import subprocess

    info = {"root": SKILL_ROOT, "git_head": None, "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", SKILL_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode == 0:
            info["git_head"] = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", SKILL_ROOT, "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
        if dirty.returncode == 0:
            info["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info
