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
    from harness.run import _load_env

    problems, info = [], []
    if not os.path.isfile(env_file):
        problems.append(
            f"env 文件不存在：{env_file}\n"
            "    修法：--env <有额度的文件>，或 export OPBENCH_ENV=..."
        )
        return problems, info
    os.environ["OPBENCH_ENV"] = env_file
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


def check_gpu(threshold: float = 20.0) -> tuple[list[str], list[str], list[int]]:
    """卡空闲 + 档位一致。返回 (problems, warns, free_cards)。"""
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
    free = [i for i, p in enumerate(pcts) if p <= threshold]
    if busy:
        detail = ", ".join(f"卡{i}:{pcts[i]:.0f}%" for i in busy)
        problems.append(
            f"有卡占用率 > {threshold}%：{detail}\n"
            "    本机共享，别抢卡。等别的会话结束"
        )
    if not free:
        problems.append(f"没有空闲卡（利用率 {pcts}）")

    uniq = sorted(set(levels))
    if len(uniq) > 1:
        warns.append(
            f"卡时钟档位不一致 {uniq}（各卡 {levels}）。\n"
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
    from harness.run import stated_budget_s, task_dir

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


def run_all(units, env_file: str, threshold: float = 20.0) -> tuple[Report, list[int]]:
    """跑全部检查。返回 (报告, 空闲卡)。"""
    rep = Report()
    arms = sorted({u.arm for u in units})

    p, info = check_env(env_file)
    rep.problems.extend(p)
    rep.info.extend(info)

    p, w, free = check_gpu(threshold)
    rep.problems.extend(p)
    rep.warns.extend(w)

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
