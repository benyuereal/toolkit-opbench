"""开跑前的全部准入检查。每一条都是踩过的坑，不是防御性编程。

## 检查项与对应的历史事故

| 检查 | 事故 |
|---|---|
| env 文件与凭据 | 三臂全 402；`ANTHROPIC_BASE_URL` 尾斜杠导致 `//v1/messages` 404 |
| 卡空闲 | 共享机器，别的会话占卡把带宽抢走，三臂被抢得不均匀 |
| 时钟档位 | 8 卡同时有 600/1200/1350，跨卡耗时不可比 |
| 臂用户隔离 | C 臂读过 A 臂的 report.md；`reference.py` 曾对三臂全可读 |
| 答案文件 | `reference.py` 逐行就是答案，644 时 agent 直接抄 |
| `--max-usd` | $5 在 ~33 分钟切断 60 分钟的题，三臂切在近似但不同的时刻 |
| budget 低于题面 | `run.py` 拒绝，白等 |
| 端口残留 | `tokenizer_worker` 占 8000，后续 run 起不来 |
| 口径自洽 | 各臂自选形状，同一"算子"差 15 倍 |

## 分级

* `problem`：阻止开跑，返回非零。
* `warn`：能跑但要知道（比如跨卡档位不一致时"题内可比、跨题不可比"）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = []
        for m in self.info:
            lines.append(f"[info] {m}")
        for m in self.warns:
            lines.append(f"[warn] {m}")
        for i, m in enumerate(self.problems, 1):
            lines.append(f"[{i}] ✗ {m}")
        return "\n".join(lines)


def check_env(env_file: str) -> tuple[list[str], list[str]]:
    """env 文件、凭据、URL 尾斜杠。返回 (problems, info)。"""
    from harness.launch import _load_env, set_env_file

    problems, info = [], []
    if not os.path.isfile(env_file):
        problems.append(
            f"env 文件不存在：{env_file}\n"
            "    修法：--env <有额度的文件>，或 export OPBENCH_ENV=..."
        )
        return problems, info
    # 必须走 set_env_file()：`ENV_FILE` 是 import 时的快照，只改 os.environ
    # 不会生效，_load_env() 仍读旧文件——那样这里的检查就查错了对象。
    set_env_file(env_file)
    env = _load_env()
    if not (env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")):
        problems.append(
            f"{env_file} 缺 ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY，三臂会全 402"
        )
    # 直接读文件原文：_load_env() 已经把尾斜杠 rstrip 了，从 dict 里看不出
    # 源文件有没有写斜杠。而"源文件有斜杠"本身是允许的（_load_env 会修），
    # 这里只做提示性检查，确认那条 rstrip 逻辑还在生效。
    url_in_file = ""
    try:
        with open(env_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("export "):
                    s = s[len("export "):].strip()
                if s.startswith("ANTHROPIC_BASE_URL="):
                    url_in_file = s.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except OSError:
        pass
    effective = env.get("ANTHROPIC_BASE_URL") or ""
    if url_in_file.endswith("/") and effective.endswith("/"):
        problems.append(
            f"ANTHROPIC_BASE_URL 尾斜杠未清除（源文件 {url_in_file!r}，"
            f"实际传给子进程 {effective!r}）→ 会拼成 //v1/messages 得 404"
        )
    elif url_in_file.endswith("/"):
        info.append(
            f"ANTHROPIC_BASE_URL 源文件带尾斜杠（{url_in_file}），"
            f"_load_env() 已修正为 {effective} ✓"
        )
    if env.get("ANTHROPIC_MODEL"):
        info.append(f"模型：{env['ANTHROPIC_MODEL']}（三臂必须同一个）")
    return problems, info


def check_gpu(threshold: float = 20.0,
              need: int = 1) -> tuple[list[str], list[str], list[int]]:
    """卡空闲 + 档位一致。返回 (problems, warns, free_cards)。

    `need` 是**至少要几张卡**。判据是"够不够用"，不是"是不是全都空着"。

    ## 为什么不能"任一卡忙就拦"

    实测（2026-11-26 09:41）：卡 0/3 被占到 67%，卡 4–7 各空 139 GB。
    三题并行只要 3 张卡，完全跑得起来，但旧判据直接把整轮拦下了。
    本机 8 张卡里长期只有 4 张可用，**"任一忙即停"等于永远跑不了**。
    """
    from bench.hw import dcu

    problems, warns = [], []
    try:
        pcts = dcu._busy_pcts()
        levels = [lv for lv, _ in dcu._sclk_levels()]
    except Exception as e:
        return [f"读不到卡状态（{type(e).__name__}: {e}）"], [], []

    if not pcts:
        return ["hy-smi -u 读不到利用率：按全忙处理"], [], []

    busy = [i for i, p in enumerate(pcts) if p > threshold]
    # 走 dcu.free_devices() 而不是在这里再写一遍 `p <= threshold`：
    # 那个判断漏掉显存，实测会把只剩 5 GB 的卡判成空闲（见 dcu 里的说明）。
    # 两处各写一遍正是这个 bug 出现两次的原因。
    free = dcu.free_devices(threshold)
    # 有卡空着但显存不够：这不一定是"别人占着"，可能是别的容器的常驻服务。
    # 单独说清楚，否则看日志的人会去等一个不会结束的会话。
    starved = [i for i, p in enumerate(pcts)
               if p <= threshold and i not in free]
    if starved:
        warns.append(
            f"卡 {starved} 没在算但可用显存 < {dcu.MIN_FREE_VRAM_GB:g} GB，已排除\n"
            "    多半是容器外租户的常驻服务占着显存。这不是本容器的进程，"
            "不要 kill，换别的卡"
        )
    if busy:
        detail = ", ".join(f"卡{i}:{pcts[i]:.0f}%" for i in busy)
        msg = (f"有卡占用率 > {threshold}%：{detail}（本机共享，不会去抢）\n"
               f"    可用 {len(free)} 张：{free}，需要 {need} 张")
        if len(free) < need:
            problems.append(msg + "\n    空闲卡不够，等别的会话结束")
        else:
            warns.append(msg + " → 够用，继续")
    elif not free:
        problems.append(f"没有既空闲又够显存的卡（利用率 {pcts}）")

    # 只在**实际要用到的那几张**卡之间比档位：卡 0 是档位 8、卡 4 是档位 1
    # 是常态（前者被别的会话压到高频，后者刚空下来还没升档），
    # 拿全机 8 张比会永远报警，而真正要派活的 4 张可能是一致的。
    uniq_used = sorted({levels[i] for i in free if i < len(levels)})
    if len(uniq_used) > 1:
        used = [levels[i] if i < len(levels) else None for i in free]
        warns.append(
            f"可用卡之间时钟档位不一致 {uniq_used}（{free} → {used}）。\n"
            "    同一题的各臂会用同一张卡 → 题内可比；**跨题**耗时不可横比。\n"
            "    修法：先 ramp 到工作档，或只挑同档位的卡"
        )
    return problems, warns, free


def check_isolation(arms: list[str]) -> tuple[list[str], list[str]]:
    """臂用户 + 实测互不可见 + 答案文件。"""
    from harness.multi import isolation as iso
    from harness.ready import check_answer_isolated

    problems, warns = [], []

    problems.extend(iso.ensure_users(arms))

    # 实测互不可见（只在 root 下做）
    problems.extend(iso.verify_isolation(arms))

    ans = check_answer_isolated()
    if ans["level"] == "fail":
        problems.append(
            f"答案文件泄漏：{ans['value']}\n    修法：{ans.get('fix_hint', '')}"
        )
    elif ans["level"] == "warn":
        warns.append(str(ans["value"]))

    legacy = [r for r in iso.scan_legacy() if r["world_readable"]]
    if legacy:
        warns.append(
            f"{len(legacy)} 个旧产物树对 others 可读（早期未隔离时产生）：\n    "
            + ", ".join(r["path"] for r in legacy[:4])
            + "\n    这些不能进新对照；跑 `--quarantine` 收紧权限"
        )
    return problems, warns


def check_residue() -> list[str]:
    """残留进程占端口。"""
    out = []
    try:
        p = subprocess.run(["pgrep", "-af", "tokenizer_worker"],
                           capture_output=True, text=True, timeout=5)
        if p.stdout.strip():
            out.append(
                "有 tokenizer_worker 残留（占 8000 端口）：\n    "
                + p.stdout.strip().replace("\n", "\n    ")
                + "\n    修法：先停服务（见 HANDOFF 第 1 节）"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def check_budgets(units) -> list[str]:
    """单元预算不能低于题面写的时间；`--max-usd` 与时间预算不能打架。"""
    from harness.launch import stated_budget_s, task_dir

    problems = []
    seen = set()
    for u in units:
        if u.task in seen:
            continue
        seen.add(u.task)
        p = os.path.join(task_dir(u.task), "task.md")
        if not os.path.isfile(p):
            problems.append(f"{u.task}: 找不到 task.md")
            continue
        with open(p, encoding="utf-8") as f:
            stated = stated_budget_s(f.read())
        if stated is not None and u.budget_s < stated:
            problems.append(
                f"{u.task}: budget {u.budget_s}s < 题面写的 {stated}s。\n"
                f"    压预算会把本该完成的样本切成 timeout，Δ 被污染"
            )
    return problems


def run_all(units, env_file: str, threshold: float = 20.0,
            parallel: bool = False) -> tuple[Report, list[int]]:
    """跑全部检查。返回 (报告, 空闲卡)。

    `parallel=True` 时至少要"一道题一张卡"（`dispatch_by_card` 的并行度
    是 min(题数, 卡数)）；串行只要一张。
    """
    rep = Report()
    arms = sorted({u.arm for u in units})
    n_tasks = len({u.task for u in units})
    need = min(n_tasks, 8) if parallel else 1

    p, info = check_env(env_file)
    rep.problems.extend(p)
    rep.info.extend(info)

    p, w, free = check_gpu(threshold, need=need)
    rep.problems.extend(p)
    rep.warns.extend(w)
    # 按可用显存从多到少排，让 --parallel-cards 先吃掉最宽裕的几张。
    # 不排的话会按编号从小到大派，稳定选中卡 0——而卡 0 常是别人占剩的那张。
    try:
        from bench.hw import dcu as _dcu
        _mem = _dcu._free_vram_gb(max(free) + 1) if free else {}
        free = sorted(free, key=lambda i: (-_mem.get(i, -1.0), i))
    except Exception:
        pass

    p, w = check_isolation(arms)
    rep.problems.extend(p)
    rep.warns.extend(w)

    rep.problems.extend(check_residue())
    rep.problems.extend(check_budgets(units))

    from harness.multi.model import check_calibration

    rep.problems.extend(check_calibration(units))

    # 口径缺失提示（judgment 题没有性能基线是正常的）
    no_cal = [u.task for u in units if u.caliber is None]
    if no_cal:
        rep.info.append(
            f"{len(set(no_cal))} 道题没有性能基线（judgment 类正常）：{sorted(set(no_cal))}"
        )
    return rep, free
