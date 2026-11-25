"""D 臂（SkillHub）：每臂一份独立副本 + 造 plugin manifest + .claude 配置。

## 与 C 臂的差别只在"给哪套知识"

  C → /workspace/toolkit_skills（我们的算子卡知识库）
  D → /workspace/SkillHub（团队共享的 44 个 Agent Skill）

两者都是**完整知识层**，不是梯度关系（B 臂才是"只有库清单"的中间臂）。

## 为什么要在副本里造 plugin.json

SkillHub 是扁平的 `skills/<name>/SKILL.md`，**没有 `.claude-plugin/plugin.json`**，
Claude Code 不认它是 plugin。实测（2026-11-24）：直接 `--plugin-dir` 指向
SkillHub 根目录，init 事件里一个 skill 都没有。

所以在副本里就地生成一个最小 manifest，把 `skills/` 声明为 plugin 的 skills 目录。
**只写进副本，不动源仓库。**
"""
import json
import os
import shutil
import subprocess

ARM = "D"
# 同 C 臂：不换用户就等于能读答案。
USER = "opb_d"
SKILLHUB_ROOT = "/workspace/SkillHub"
LINK_NAME = "skillhub"
# 副本里的 plugin 根：cwd/skillhub/.claude-plugin/plugin.json + cwd/skillhub/skills/
PLUGIN_SUBDIR = LINK_NAME

# 中性说明：只说明"是什么、入口在哪、怎么调"。类别说明（如"有 kernel 优化
# loop 类 skill"）合格；具体算子名 / 具体 skill 名 / 具体文件路径一律不写。
# —— 业主 2026-11-24 判据，锁定此版本，不再微调。
CLAUDE_MD = """# 本机已安装的 skillhub 插件

这是一套 Agent Skills 插件，用 `Skill` 工具加载，参数格式 `skillhub:<名字>`。
skill 都在 `{plugin}/skills/` 下，每个子目录是一个 skill，目录里的 SKILL.md 描述它做什么、怎么用。
其中有做 A/B 实验 + 判定的 kernel 优化 loop 类 skill，按需加载。

**加载哪个 skill、优化哪个算子、怎么实现，由你根据任务自行判断**——
先按需查看 `skills/` 下有哪些 skill，再决定加载哪个。不要凭记忆假设。

## 本机事实

gfx938 DCU。时钟/占用查 `hy-smi`（**不是** rocprof / nvidia-smi / K100）。
"""


def build_prompt(task_md: str) -> str:
    """与 A 臂逐字相同的题面。skill 由 cwd 的 .claude 配置自动进入。"""
    return task_md


def _make_plugin_manifest(dst: str) -> None:
    """在副本里造一个最小 plugin manifest。

    Claude Code 的约定：plugin 根下 `.claude-plugin/plugin.json`，
    skills 在 plugin 根的 `skills/<name>/SKILL.md`。
    SkillHub 已经是 `skills/<name>/SKILL.md`，所以只需要补 manifest。
    """
    d = os.path.join(dst, ".claude-plugin")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "plugin.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "skillhub",
            "version": "1.0.0",
            "description": "SkillHub — 团队共享 Agent Skills（44 个，含 kernel 优化 loop）",
        }, f, ensure_ascii=False, indent=2)


def stage_files(cwd: str) -> list[str]:
    """拷 SkillHub 进 cwd，造 plugin manifest + .claude 配置 + CLAUDE.md。"""
    if not os.path.isdir(SKILLHUB_ROOT):
        raise FileNotFoundError(
            f"SkillHub 不存在：{SKILLHUB_ROOT}。缺它 D 臂就退化成了 A 臂"
        )

    dst = os.path.join(cwd, LINK_NAME)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(
        SKILLHUB_ROOT, dst,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        symlinks=True,
    )
    _make_plugin_manifest(dst)

    staged = [LINK_NAME]

    claude_dir = os.path.join(cwd, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    settings = os.path.join(claude_dir, "settings.local.json")
    # marketplace 名与源仓库的 marketplace 不同，避免与 C 臂冲突。
    # 这里给的是**目录形式的 marketplace**，指向 cwd 内的副本。
    with open(settings, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "enabledPlugins": {"skillhub": True},
            "extraKnownMarketplaces": {},
        }, ensure_ascii=False, indent=2) + "\n")
    staged.append(os.path.relpath(settings, cwd))
    import subprocess as _sp
    _sp.run(["chown", "-R", "opb_d:opb_d", cwd], check=False)
    _sp.run(["chmod", "-R", "u+rwX,go-rwx", cwd], check=False)

    with open(os.path.join(cwd, "CLAUDE.md"), "w", encoding="utf-8") as f:
        f.write(CLAUDE_MD.format(plugin=PLUGIN_SUBDIR))
    staged.append("CLAUDE.md")
    return staged


def mount(cwd: str) -> list[str]:
    return []


def skill_version() -> dict:
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
