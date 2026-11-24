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
SKILLHUB_ROOT = "/workspace/SkillHub"
LINK_NAME = "skillhub"
# 副本里的 plugin 根：cwd/skillhub/.claude-plugin/plugin.json + cwd/skillhub/skills/
PLUGIN_SUBDIR = LINK_NAME

# 只列名字和一句话用途，不指定"这个任务用哪个"。
# 清单来自 SkillHub/README.md 的 catalog（44 个里挑与本项目相关的）。
CLAUDE_MD = """# 本机已安装的 skillhub 插件

用 `Skill` 工具调用，参数格式 `skillhub:<名字>`。
全部 skill 在 `{plugin}/skills/<名字>/`，每个目录有 SKILL.md。

## 与本项目相关的

kernel 优化：
  lightop-kernel-agent-loop（**自主优化 LightOp 融合算子的 loop**）
  lightop-kernel-knowledge · kernel-hip-optimize · kernel-hip-baseline
  kernel-hip-handbook · kernel-triton-optimize · kernel-triton-baseline
  kernel-model-triage

调优 / 剖析：
  gemm-tuning · hcu-aiter-moe-tuning · dcu-profiler-report
  profile-hipprof-analysis · llm-torch-profiler-analysis
  triton-dcu-profiler-report · sglang-sota-humanize-loop

环境 / 排查：
  hy-smi · dcu-coredump-vmfault · hcu-monitor · hcu-cumask

按任务需要**自行判断**该加载哪个。

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
