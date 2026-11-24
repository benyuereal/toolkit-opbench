"""C 臂（toolkit-skills）：每个臂一份独立副本 + 独立 Claude 配置。

## 为什么要拷贝而不是软链

`loop.py` 把状态写到 `<卡片>/runs/loop/*.jsonl`。如果软链到共享的
`/workspace/toolkit_skills`，那么：
  * 三个臂会**互相污染**同一份 runs/loop/ 流水
  * 跑完会**改写源仓库**（被测对象被测试改动，靶子就变了）
  * 并发跑时两个臂会同时写同一个 jsonl

所以每臂拷贝一份（71 MB / 0.6 s，代价可接受），并把副本路径记进 trace。

## 为什么还要写 .claude/settings.local.json

只把目录拷进 cwd 是不够的——Claude Code 不会自动发现它是一个 plugin。
实测（2026-11-24）：
  * 只装插件、无 CLAUDE.md → agent 从不调用 Skill 工具，去 `find /` 全盘搜
  * 装插件 + CLAUDE.md 列出 skill 清单 → agent 自己选对了 skill

`settings.local.json` 启用 plugin，`CLAUDE.md` 列清单；两者缺一不可。
"""
import os
import shutil
import subprocess

ARM = "C"
SKILL_ROOT = "/workspace/toolkit_skills"
# 副本放在 cwd 下的这个名字。CLAUDE.md 与 prompt 都引用它。
LINK_NAME = "skill"
PLUGIN_SUBDIR = os.path.join(LINK_NAME, "Plugins", "ops")

# 落在 cwd 的 CLAUDE.md：只列清单 + 用法，**不指定"这个任务该用哪个"**。
# 指定了就变成提示，测不出 agent 自己找 skill 的能力。
#
# 实测依据（2026-11-24）：
#   只写"已安装 ops 插件，自己看"     → agent 跑 `claude plugin details`，
#                                      但 --plugin-dir 装的 plugin 查不到 → 放弃
#   列出 16 个 skill 名 + 不指定用途  → agent 自己选了 op-roofline，回答引用了 skill 内容 ✅
CLAUDE_MD = """# 本机已安装的 ops skill 插件（16 个）

用 `Skill` 工具调用，参数格式 `ops:<名字>`。

算子：{operator_skills}

方法：{method_skills}

按任务需要**自行判断**该加载哪个。

## 优化闭环

如果被优化的算子有对应的卡（`{plugin}/skills/op-<算子名>/`），
用它自带的循环引擎推进，不要只测一次就下结论：

    python3 -I {plugin}/engine/loop.py -C {plugin}/skills/op-<算子名> status
    python3 -I {plugin}/engine/loop.py -C {plugin}/skills/op-<算子名> queue
    python3 -I {plugin}/engine/loop.py -C {plugin}/skills/op-<算子名> run <tag>

`run` 会调卡自带的 bench/ab.py 做 A/B、按 done_when 外部裁决，
不接受"我觉得快了"。

## 本机事实

gfx938 DCU。时钟/占用查 `hy-smi`（**不是** rocprof / nvidia-smi / K100）。
"""

OPERATOR_SKILLS = (
    "op-fused-add-rmsnorm · op-rmsnorm-fused · op-moe-topk-gating · op-moe-w8a8\n"
    "      op-moe-align-permute · op-gemm-fp8 · op-sparse-mla"
)
METHOD_SKILLS = (
    "op-roofline（带宽/峰值分母）· op-bench（微基准）· op-contract · op-gate\n"
    "      op-optimize · op-profiling · op-hip-compile · op-source-patch · op-stack-design"
)


def build_prompt(task_md: str) -> str:
    """prompt 与 A 臂**完全相同**（只有题面）。

    skill 通过 cwd 里的 .claude 配置与 CLAUDE.md 自动进入上下文，
    不靠 prompt 提示——这样三臂的 prompt 逐字相同，
    差出来的就是"有没有那个环境"，而不是"提示写了什么"。
    """
    return task_md


def stage_files(cwd: str) -> list[str]:
    """把 skill 拷进 cwd，并写好 .claude 配置 + CLAUDE.md。"""
    if not os.path.isdir(SKILL_ROOT):
        raise FileNotFoundError(
            f"C 臂的知识库不存在：{SKILL_ROOT}。缺它 C 臂就退化成了 A 臂"
        )

    dst = os.path.join(cwd, LINK_NAME)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    # 不拷 .git：省 26 MB，且避免副本被误当成独立仓库。
    shutil.copytree(
        SKILL_ROOT, dst,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "runs"),
        symlinks=True,
    )

    staged = [LINK_NAME]

    # 启用 plugin（Claude Code 只认 settings 里的 enabledPlugins）
    claude_dir = os.path.join(cwd, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    settings = os.path.join(claude_dir, "settings.local.json")
    with open(settings, "w", encoding="utf-8") as f:
        f.write('{\n  "enabledPlugins": {"ops@dcu-toolkit-marketplace": true}\n}\n')
    staged.append(os.path.relpath(settings, cwd))

    # CLAUDE.md：列清单 + 用法，不指定用途
    md = CLAUDE_MD.format(plugin=PLUGIN_SUBDIR, operator_skills=OPERATOR_SKILLS,
                          method_skills=METHOD_SKILLS)
    with open(os.path.join(cwd, "CLAUDE.md"), "w", encoding="utf-8") as f:
        f.write(md)
    staged.append("CLAUDE.md")
    return staged


def mount(cwd: str) -> list[str]:
    """无额外挂载：skill 已由 stage_files 拷进 cwd。

    保留这个函数是因为执行器按 `hasattr(mod, "mount")` 决定要不要调。
    """
    return []


def skill_version() -> dict:
    """记录被测 skill 的版本。离开它结果没法解释。"""
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
