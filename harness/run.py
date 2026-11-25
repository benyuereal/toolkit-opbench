"""单臂单题执行器：隔离 / 环境注入 / 超时 / 产物收集。

执行契约（与 protocol.md、docs/workflow.md 第 3 步一致）：
  1. 建 runs/<arm>/<task>/attempt-N/ 作为 cwd，chmod 777（A 臂要以 opbench 写）
  2. 注入上下文（A 无 / B 只给 libraries-registry.yaml / C 给 skill 仓库）
  3. L5 不碰 GPU；需要 GPU 的题：持锁 → 确认空闲 → ramp → 记 sclk
  4. timeout <budget> claude -p "<prompt>" --output-format json
  5. 收集 report.md / scripts/ / results/
  6. 写 trace.jsonl：wall time、tokens、sclk、exit code
超时即判失败，不续跑——补跑会破坏固定预算的可比性。

## 三个实测决定（2026-11-23，claude 2.1.330）

* **`--allowedTools` 白名单，不用 `--bypassPermissions`。** 后者被安全策略
  拒绝，而且本来就不该用：三臂必须跑在完全相同的工具条件下，白名单是
  实验配置，关掉审批不是。`--permission-mode acceptEdits` 单独用不够——
  实测 Bash 仍会被拒（`permission_denials` 里有 Bash），agent 会写出一份
  「我没能运行」的报告，那就测成了"谁更会找理由"。
* **prompt 走 stdin，不走 argv。** 题面里有引号、反引号、中文；
  拼进 shell 字符串迟早出错，而错一次就是一轮作废的样本。
* **`< /dev/null`。** 不给 stdin 时 claude 会等 3 秒并打一条 Warning，
  3 秒 × 18 次实验白等。

## 预算

wall time 用 `timeout` 硬切；token 用 `--max-budget-usd` 兜底（CLI 没有
token 上限开关，只有美元预算）。两者任一触顶都记 `timeout`/`budget`，
**不续跑**。超时后子进程组整体杀掉，避免 agent 起的长跑进程留到下一轮。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(ROOT, "tasks")
RUNS_DIR = os.path.join(ROOT, "runs")
BASELINES_DIR = os.path.join(ROOT, "baselines")
TRACE_PATH = os.path.join(RUNS_DIR, "trace.jsonl")
CLAUDE = "/opt/opbench-node/bin/claude"
# 被测 agent 连 API 用的环境文件。默认 tosc.env（qwen3.8-27b，实测有额度）；
# 三个 opendesign*.env 全 402。三臂必须用同一个，否则测出的是模型差，不是 skill 差。
# 模型换端点时用 OPBENCH_ENV 覆盖。
ENV_FILE = os.environ.get("OPBENCH_ENV", "/workspace/tosc.env")
# 某些端点（qwen 的 OpenCode 网关）只接受有限的 reasoning effort 档位，
# CLI 默认的 high 会被 400 拒掉。用 OPBENCH_EFFORT 传，空则不传该 flag。
EFFORT = os.environ.get("OPBENCH_EFFORT", "")

# 三臂共用的工具白名单。改它等于改实验条件，三臂必须同步。
ALLOWED_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite")
# 默认**不设**美元上限（None）。实测 5.0 会在 33 分钟切断三臂，
# 把 3600s 的时间预算架空。要设就显式传 --max-usd。
DEFAULT_MAX_USD = None

# 产物里允许/应该出现的顶层条目。其余一律视为 agent 自造，
# 记录但不影响判分（判分只认 report.md/scripts/results/solution.json）。
ARTIFACT_NAMES = ("report.md", "scripts", "results", "solution.json")

# 需要 GPU 的 tier。**L5 不是一刀切不需要 GPU**：
# 判定题里有一半要实测（fp8 原生性、双流重叠度），另一半只要查符号
# （hadamard symbol、all-reduce 归属）。按 tier 判会把前者误标成无 GPU，
# 于是绕过占用检查直接在忙卡上跑。
GPU_TIERS = ("L1", "L2", "L3", "L4")


def needs_gpu(defn) -> bool:
    """tier 之外再看题面声明：definition 里显式写了 needs_gpu 就以它为准。"""
    explicit = (defn.raw or {}).get("needs_gpu")
    if explicit is not None:
        return bool(explicit)
    return defn.tier in GPU_TIERS


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _arm_module(arm: str):
    from harness.arms import a, b, c, skillhub

    return {"A": a, "B": b, "C": c, "D": skillhub}[arm.upper()]


def task_dir(task_id: str) -> str:
    """按 id 找题目录。

    目录名与 definition.id 不总相同（如 tasks/l5-asm-fmoe-a8 的 id 是
    moe-asm-fmoe-l5）。两处都认，但必须唯一命中——命中两处就报错，
    因为"跑的是哪道题"不能靠猜。
    """
    direct = os.path.join(TASKS_DIR, task_id)
    if os.path.isdir(direct):
        return direct
    matches = []
    for name in sorted(os.listdir(TASKS_DIR)):
        d = os.path.join(TASKS_DIR, name)
        if name.startswith("_") or not os.path.isdir(d):
            continue
        defn = os.path.join(d, "definition.json")
        if not os.path.isfile(defn):
            continue
        try:
            with open(defn, encoding="utf-8") as f:
                if json.load(f).get("id") == task_id:
                    matches.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"id {task_id!r} 命中多个目录 {matches}，拒绝猜")
    raise FileNotFoundError(
        f"找不到题目 {task_id!r}（既不是目录名，也不是任何 definition.id）"
    )


def load_definition(td: str):
    from bench.data import definition as defn_mod

    return defn_mod.load(os.path.join(td, "definition.json"))


def _next_attempt(base: str) -> tuple[int, str]:
    n = 1
    while os.path.exists(os.path.join(base, f"attempt-{n}")):
        n += 1
    return n, os.path.join(base, f"attempt-{n}")


def prepare_cwd(task_id: str, arm: str, dry_run: bool = False) -> str:
    """建 runs/<arm>/<task>/attempt-N/ 并 chmod 777。返回 cwd。

    目录名一律用 **definition.id**（不是调用时传的名字）。原因：目录名与
    id 不一定相同（tasks/l5-asm-fmoe-a8 的 id 是 moe-asm-fmoe-l5），
    若按传入名建目录，同一道题在不同臂下会落到两个目录，汇总时被当成
    两道题——三臂配对就断了，Δ 永远算不出来。

    attempt-N 而不是直接覆盖：失败样本比成功样本信息量大，但复跑会
    破坏"固定预算"的可比性。所以保留每一次，且下游只认 attempt-1
    （见 select_attempt）——不续跑，但留证据。

    dry_run 落到 `_dry-run/`，**不占 attempt 号**：一次预演若占掉
    attempt-1，真正的那次就变成 attempt-2、被下游忽略，等于白跑。
    """
    if dry_run:
        cwd = os.path.join(RUNS_DIR, f"arm-{arm.upper()}", task_id, "_dry-run")
        os.makedirs(cwd, exist_ok=True)
        os.chmod(cwd, 0o777)
        return cwd

    base = os.path.join(RUNS_DIR, f"arm-{arm.upper()}", task_id)
    os.makedirs(base, exist_ok=True)
    n, cwd = _next_attempt(base)
    os.makedirs(cwd)
    os.chmod(cwd, 0o777)
    # 题面要求脚本放 scripts/、原始输出放 results/。
    # Claude 的 Write 不自动建父目录 —— 实测 C 臂写 scripts/x.py 失败后反复查权限跑偏。
    for _sub in ("scripts", "results"):
        os.makedirs(os.path.join(cwd, _sub), exist_ok=True)
    return cwd


# A 臂的 cwd 必须搬到仓库之外：cwd 在仓库内时，agent 顺着 `..` 就能
# 读到 expect.md（答案）、definition.json（含 expected_verdict）、
# 以及 B/C 臂的产物。实测 opbench 用户对仓库全部可读。
ISOLATED_ROOT = "/tmp/opbench-run"



def _claim_for_arm(arm: str, path: str) -> None:
    """把 path 及其下两级交给该臂的用户，并设为 700。"""
    if os.geteuid() != 0:
        return
    user = getattr(_arm_module(arm), "USER", None)
    if not user:
        return
    try:
        import pwd
        pw = pwd.getpwnam(user)
    except (ImportError, KeyError):
        return
    targets = [path]
    try:
        for task in os.listdir(path):
            tp = os.path.join(path, task)
            targets.append(tp)
            for sub in os.listdir(tp):
                targets.append(os.path.join(tp, sub))
    except OSError:
        pass
    for t in targets:
        try:
            os.chown(t, pw.pw_uid, pw.pw_gid)
            os.chmod(t, 0o700)
        except OSError:
            pass


def isolated_cwd(task_id: str, arm: str, dry_run: bool = False) -> str:
    """A 臂用：仓库之外的 cwd。语义与 prepare_cwd 一致（attempt-N、dry-run）。"""
    if dry_run:
        cwd = os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}", task_id, "_dry-run")
        os.makedirs(cwd, exist_ok=True)
        os.chmod(cwd, 0o777)
        return cwd
    base = os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}", task_id)
    os.makedirs(base, exist_ok=True)
    n, cwd = _next_attempt(base)
    os.makedirs(cwd)
    os.chmod(cwd, 0o777)
    # 题面要求脚本放 scripts/、原始输出放 results/。
    # Claude 的 Write 不自动建父目录 —— 实测 C 臂写 scripts/x.py 失败后反复查权限跑偏。
    for _sub in ("scripts", "results"):
        os.makedirs(os.path.join(cwd, _sub), exist_ok=True)
    # 一臂一用户：把 arm-X 整棵树交给该臂用户并 700，别的臂（另一个用户）就进不来。
    # 只换用户不够——三个臂同用 opbench 时，它们仍能互读对方的 cwd（实测 C 读了 arm-A 的 report.md）。
    _claim_for_arm(arm, os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}"))
    return cwd


def _isolated_arm(arm: str) -> bool:
    """有独立用户就说明这条臂要隔离（当前只有 A）。"""
    return bool(getattr(_arm_module(arm), "USER", None))


def guard_answer_readable(task_id: str, arm: str) -> None:
    """跑之前先确认：这条臂**读不到**答案侧文件。

    这不是"防小人"的加固，而是**实验前提**：能读到答案的臂不能被当成
    基线。与其事后靠指纹猜它有没有看，不如开跑前就确认它看不到。
    读得到就直接拒绝——把问题暴露在数据产生之前。
    """
    if not _isolated_arm(arm):
        return
    from harness.arms.a import USER

    targets = [
        os.path.join(task_dir(task_id), "expect.md"),
        os.path.join(task_dir(task_id), "definition.json"),
        # reference.py 是**正确答案本身**（逐行可执行的 torch 参考实现），
        # 读到它比读到 expect.md 更省事：直接抄 linalg 语义，连猜都不用猜。
        # 之前只挡了前两个，实测 opbench 用户对它是可读的（644）。
        os.path.join(task_dir(task_id), "reference.py"),
    ]
    readable = []
    for p in targets:
        if not os.path.isfile(p):
            continue
        probe = f'test -r {_quote_for_shell([p])}'
        r = subprocess.run(["su", "-", USER, "-c", probe], capture_output=True)
        if r.returncode == 0:
            readable.append(os.path.relpath(p, ROOT))
    if readable:
        raise PermissionError(
            f"{arm} 臂能读到 {readable}。这是实验前提问题，不是小疏漏："
            "能读到答案的臂不能当基线。修法（任一）："
            f"chmod 600 {os.path.relpath(task_dir(task_id), ROOT)}/expect.md "
            f"与 definition.json；或让 {arm} 臂在仓库外的 cwd 里跑"
            "（见 harness.run.isolated_cwd）"
        )


def select_attempt(base: str) -> str | None:
    """下游评分只看 attempt-1。没有 attempt-1 就没有这一轮的结果。"""
    first = os.path.join(base, "attempt-1")
    return first if os.path.isdir(first) else None


def build_cmd(task_dir: str, budget_s: int, max_usd: float | None = DEFAULT_MAX_USD,
              plugin_dir: str | None = None) -> list[str]:
    """组命令。prompt 由 stdin 给，所以这里不含题面。

    模型与 effort 从 OPBENCH_* 环境变量取（见模块顶部），三臂一致。

    `plugin_dir`：该臂 cwd 里的 plugin 根。**每个臂指向自己的副本**，
    这样 skill 通过 Claude Code 的原生机制进入上下文，不靠 prompt 提示，
    也不会串到别的臂。
    """
    cmd = [
        CLAUDE,
        "-p",
        "--output-format", "json",
        "--allowedTools", *ALLOWED_TOOLS,
        # Claude Code 把 .py 判成 sensitive file，-p 非交互模式下无法确认 →
        # 写自研 kernel 直接被拒（实测 C 臂只能 Write 到 scripts/ 再 cp 绕过去）。
        # 隔离边界由 Unix 权限保证（700 + 一臂一用户），不靠这个开关。
        "--dangerously-skip-permissions",
    ]
    # 美元上限只在显式给了正数时才传。**默认不设**：实测 --max-usd 5.0
    # 会在 33 分钟就把三臂全部切断（A $4.89 / C $5.01 / D $4.98），
    # 把 3600s 的时间预算架空，C 只差 0.01 美元就判 error —— 样本被
    # 美元切掉而不是被能力切掉，Δ 不可用。预算约束只留 wall time。
    if max_usd is not None and max_usd > 0:
        cmd += ["--max-budget-usd", f"{max_usd:.2f}"]
    if EFFORT:
        cmd += ["--effort", EFFORT]
    model = os.environ.get("OPBENCH_MODEL", "")
    if model:
        cmd += ["--model", model]
    if plugin_dir:
        cmd += ["--plugin-dir", plugin_dir]
    return cmd


def _load_env() -> dict:
    """读 ENV_FILE（KEY=VALUE），返回 dict。走 subprocess 的 env= 传进去，
    不走 shell source，所以不受 export 影响。

    **坑（业主实测）：tosc.env 的 `ANTHROPIC_BASE_URL` 结尾带 `/`。**
    原样传进去，Claude Code 会拼成 `...4001//v1/messages` → 404。
    所以 URL 类 key 统一 rstrip("/")。
    """
    env = {}
    if not os.path.isfile(ENV_FILE):
        return env
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            val = val.strip().strip('"').strip("'")
            # URL 类 key 去尾斜杠，避免拼出 //v1/... → 404
            if key.endswith("URL") or key.endswith("_URL"):
                val = val.rstrip("/")
            env[key] = val
    return env


def _su_prefix(arm: str) -> list[str]:
    """A 臂换用户。其余臂保持当前用户（C 臂要读 skill）。"""
    mod = _arm_module(arm)
    user = getattr(mod, "USER", None)
    if not user:
        return []
    if os.geteuid() == 0:
        return ["su", "-", user, "-c"]
    # 非 root 时无法切换用户；静默继续会把 A 臂跑成非隔离，必须硬失败
    import getpass

    if getpass.getuser() != user:
        raise PermissionError(
            f"A 臂要求以 {user} 用户运行（隔离读不到 skill），"
            f"但当前是 {getpass.getuser()} 且不是 root，无法切换。"
            "以 root 运行执行器，或手工用 su - {user} 起。"
        )
    return []


def _quote_for_shell(argv: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


# 题面自己写明的预算（"时间预算：30 分钟"）。低于它的预算会切掉一个
# 本来能完成的样本，测出来的是"我们给的时间不够"而不是"agent 不行"。
# 这类样本会污染 Δ，所以直接拒绝，而不是跑完再解释。
_STATED_BUDGET = re.compile(r"时间预算[：:]\s*\**\s*(\d+)\s*分钟")


def stated_budget_s(task_md: str) -> int | None:
    m = _STATED_BUDGET.search(task_md)
    return int(m.group(1)) * 60 if m else None


# 各臂的 attempt 目录可能落在两处：仓库内 runs/（B/C/D 默认）与
# /tmp/opbench-run/（A 臂与一臂一用户隔离后所有臂）。守卫必须两处都查，
# 否则"只扫 runs/"时守卫形同虚设（实测三臂在 /tmp 下，守卫一次都没拦过）。
def _guard_roots() -> tuple[str, ...]:
    """每次调用时重算，测试可以 patch RUNS_DIR 或 ISOLATED_ROOT 生效。"""
    iso = globals().get("ISOLATED_ROOT", "/tmp/opbench-run")
    return (RUNS_DIR, iso)


def _meta_of(arm: str, task_id: str) -> dict | None:
    for root in _guard_roots():
        p = os.path.join(root, f"arm-{arm}", task_id, "attempt-1", "meta.json")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
    return None


def _ended_naturally(arm: str, task_id: str) -> bool | None:
    """该臂上一轮是自然结束，还是被预算/超时切断。

    只有"自然结束"的样本才算真的用到了它的预算上限；被切断的样本
    说明上限本身成了约束。重跑一个被切断的臂时不必强求同一上限——
    否则永远无法把那个样本做完。

    返回 **None** 表示无法判定（旧记录没有 `_stdout.jsonl`，
    或文件里没有 result 行）。调用方必须把 None 当成"不可比"报出来，
    **不能静默放行**——否则早期记录会绕过整个预算守卫。
    """
    meta = _meta_of(arm, task_id)
    if not meta:
        return None
    if meta.get("status") == "timeout":
        return False
    for root in _guard_roots():
        p = os.path.join(root, f"arm-{arm}", task_id, "attempt-1", "_stdout.jsonl")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    lines = [l for l in f if l.strip().startswith("{")]
                if not lines:
                    return None
                last = __import__("json").loads(lines[-1])
                return last.get("terminal_reason") in (None, "completed")
            except (OSError, ValueError):
                return None
    return None


def assert_arms_comparable(task_id: str, budget_s: int, max_usd: float | None) -> list[str]:
    """同一道题的已有臂必须与新这一轮用同一组预算，否则拒绝开跑。

    三臂对照的全部意义是"只有上下文不同"。预算（wall time 与美元上限）
    只要有一项不同，测出来的就是预算差，不是 skill 差——而且这种差异
    会伪装成结论，比明显报错危险得多。

    **例外**：上一轮被预算切断的臂（terminal_reason == budget_exhausted /
    status == timeout），重跑时允许换预算——那个样本本来就不完整，
    强求同一上限等于永远修不好它。
    """
    problems = []
    for arm in ("A", "B", "C", "D"):
        meta = _meta_of(arm, task_id)
        if not meta:
            continue
        prev_budget = meta.get("budget_s")
        if prev_budget is not None and int(prev_budget) != int(budget_s):
            problems.append(
                f"arm-{arm} 用的是 budget={prev_budget}s，这次是 {budget_s}s"
            )
        prev_usd = meta.get("max_usd")
        natural = _ended_naturally(arm, task_id)
        if natural is None:
            # 无法判定 = 旧记录缺 _stdout.jsonl。不能当"自然结束"放行，
            # 也不能当"被切断"跳过——两种都会让守卫静默失效。
            problems.append(
                f"arm-{arm} 的 attempt-1 缺 _stdout.jsonl，"
                "无法判断上一轮是否被预算切断；删掉该 attempt 再重跑"
            )
            continue
        if not natural:
            # 被切断的臂不参与美元比对
            continue
        if prev_usd is None and meta.get("max_usd_backfilled") is not True:
            if max_usd is not None:
                # 旧记录没记美元上限、这次又设了 → 无法核对，报出来
                problems.append(
                    f"arm-{arm} 的 meta 没有记录 max_usd，而这次设了 {max_usd}"
                )
            continue
        if prev_usd is not None and max_usd is not None:
            if abs(float(prev_usd) - float(max_usd)) > 1e-9:
                problems.append(
                    f"arm-{arm} 用的是 max_usd={prev_usd}，这次是 {max_usd}"
                )
        elif prev_usd is not None and max_usd is None:
            # 上一轮有上限、这轮取消：只有上一轮利用率远低于上限才算可比
            mu = meta.get("model_usage") or {}
            cost = sum((v.get("costUSD") or 0) for v in mu.values())
            if cost >= float(prev_usd) * 0.95:
                problems.append(
                    f"arm-{arm} 上一轮花到 ${cost:.2f}/{prev_usd}（贴近上限），"
                    "取消上限会让它跑得比别的臂久，不可比"
                )
    if problems:
        raise ValueError(
            f"{task_id}: 三臂预算不一致，拒绝开跑（{'；'.join(problems)}）。"
            "要换预算就三臂一起重跑：删掉该题的 runs/arm-*/<task> 再跑"
        )
    return problems


def run_one(task_id: str, arm: str, budget_s: int,
            max_usd: float | None = DEFAULT_MAX_USD, dry_run: bool = False) -> dict:
    arm = arm.upper()
    td = task_dir(task_id)
    defn = load_definition(td)
    # 统一用 definition.id 作为一等标识：目录名可能只是它的旧称。
    task_id = defn.id
    mod = _arm_module(arm)

    with open(os.path.join(td, "task.md"), encoding="utf-8") as f:
        task_md = f.read()
    stated = stated_budget_s(task_md)
    if stated is not None and budget_s < stated:
        raise ValueError(
            f"{task_id}: --budget {budget_s}s 低于题面写明的 {stated}s。"
            "压预算会把本来能完成的样本切成 timeout，Δ 就被污染了。"
            f"要么给足 {stated}s，要么改题面的预算说明（改题面要重新记哈希）"
        )
    prompt = mod.build_prompt(task_md)
    # 测量口径由 harness 统一下发，三臂逐字相同。
    # 不这么做的话各臂会自选 axes，测出来的根本不是同一个算子
    # （实测 A 臂 SKV=16384/TOPK=2048 vs C/D 8192/256，耗时差 15x）。
    from harness import measure_spec
    _spec = measure_spec.render(task_id)
    if _spec:
        prompt = prompt.rstrip() + _spec

    from bench.check import integrity

    task_md_sha256 = integrity.hash_file(os.path.join(td, "task.md"))

    # 已有臂的预算必须与这一轮一致。dry_run 也检查：先看清再跑。
    assert_arms_comparable(task_id, budget_s, max_usd)

    # 先确认这条臂读不到答案，再建目录——把问题挡在数据产生之前。
    guard_answer_readable(task_id, arm)

    # 有独立用户的臂（A）把 cwd 放到仓库之外，`..` 摸不到 expect.md。
    if _isolated_arm(arm):
        cwd = isolated_cwd(task_id, arm, dry_run=dry_run)
    else:
        cwd = prepare_cwd(task_id, arm, dry_run=dry_run)
    staged = list(mod.stage_files(cwd) or [])
    mounted = list(mod.mount(cwd) or []) if hasattr(mod, "mount") else []

    # 该臂的 plugin 根（cwd 内副本）。只有声明了 PLUGIN_SUBDIR 的臂才有。
    # **必须用 cwd 里的副本**，不能用共享源：loop.py 会往 <卡片>/runs/loop/ 写状态，
    # 共享源会让三个臂互相污染，还会改写被测对象本身。
    plugin_sub = getattr(mod, "PLUGIN_SUBDIR", None)
    plugin_dir = os.path.join(cwd, plugin_sub) if plugin_sub else None

    record = {
        "task": task_id,
        "task_dir": os.path.relpath(td, ROOT),
        "arm": arm,
        "cwd": os.path.relpath(cwd, ROOT),
        "budget_s": budget_s,
        "max_usd": max_usd,
        "started_at": _now(),
        "task_md_sha256": task_md_sha256,
        "staged": staged,
        "mounted": mounted,
        "needs_gpu": needs_gpu(defn),
        "sclk_mhz": None,
        "sclk_end_mhz": None,
        "wall_time_s": None,
        "exit_code": None,
        "status": None,
        "tokens": None,
        "num_turns": None,
        "permission_denials": None,
        "artifacts": [],
        "stdout_bytes": 0,
        "stderr_bytes": 0,
    }

    # 有 skill_version() 的臂（C=toolkit_skills, D=SkillHub）都要记版本。
    # 不记的话"这一轮看的是哪个版本的知识库"无从追溯，跨轮比较不成立。
    if hasattr(mod, "skill_version"):
        record["skill_version"] = mod.skill_version()

    if dry_run:
        record["status"] = "dry-run"
        record["prompt_sha256"] = _sha(prompt)
        record["cmd"] = _su_prefix(arm) + [_quote_for_shell(
            build_cmd(td, budget_s, max_usd, plugin_dir=plugin_dir))]
        _write_attempt_meta(cwd, record, prompt)
        return record

    # ---- GPU 段：持锁 + 空闲检查 + ramp（只有需要 GPU 的题）----
    pinned = os.environ.get("OPBENCH_DEVICE", "")  # 单卡绑定时由调度器设
    if record["needs_gpu"]:
        from bench.hw import dcu

        if pinned:
            # 绑卡并行模式：调度器已经把这个进程用 CUDA_VISIBLE_DEVICES
            # 绑到一张它认为空闲的卡上。此时 hy-smi 仍报告全部 8 张卡，
            # is_busy()/pick_device() 会看到别的卡忙而误判"整机忙"。
            # 所以这里只做 ramp + 记录，把"这台绑定的卡忙不忙"交给调用方
            # （它选卡时已查过）。锁仍然要拿——同一张卡上不许两个单元叠。
            # 每卡一把锁：同一张卡上不许叠两个单元，但不同卡可以并行。
            with dcu.gpu_lock(path=f"/tmp/opbench-gpu-card{pinned}.lock",
                              wait_s=900, poll_s=10) as lock:
                if not lock["acquired"]:
                    record["status"] = "gpu_locked"
                    record["note"] = (f"绑卡 {pinned} 但该卡锁被占"
                                      f"（holder pid={lock.get('holder_pid')}），本轮不跑")
                    _write_attempt_meta(cwd, record, prompt)
                    return record
                record["device"] = int(pinned)
                record["pinned"] = True
                record["gpu_visible_env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                record["sclk_mhz"] = dcu.ramp_to_work_clock()
                record["status"], record["exit_code"], record["wall_time_s"] = _execute(
                    cwd, prompt, arm, budget_s, max_usd, record, plugin_dir=plugin_dir,
                )
        else:
            # 串行模式：等锁 + 挑空闲卡（原逻辑）
            with dcu.gpu_lock(wait_s=900, poll_s=10) as lock:
                if not lock["acquired"]:
                    record["status"] = "gpu_locked"
                    record["note"] = (f"等了 {lock.get('waited_s')}s 仍被持有"
                                      f"（holder pid={lock.get('holder_pid')}），本轮不跑")
                    _write_attempt_meta(cwd, record, prompt)
                    return record
                dev = dcu.pick_device()
                if dev is None:
                    record["status"] = "gpu_busy"
                    record["note"] = f"没有空闲卡：{dcu._busy_pcts()}"
                    _write_attempt_meta(cwd, record, prompt)
                    return record
                record["device"] = dev
                record["gpu_visible_env"] = os.environ.get("HIP_VISIBLE_DEVICES")
                record["sclk_mhz"] = dcu.ramp_to_work_clock()
                record["status"], record["exit_code"], record["wall_time_s"] = _execute(
                    cwd, prompt, arm, budget_s, max_usd, record,
                    plugin_dir=plugin_dir,
                )
            record["sclk_end_mhz"] = dcu.current_sclk()
    else:
        record["status"], record["exit_code"], record["wall_time_s"] = _execute(
            cwd, prompt, arm, budget_s, max_usd, record,
            plugin_dir=plugin_dir,
        )

    record["artifacts"] = _collect_artifacts(cwd)
    _write_attempt_meta(cwd, record, prompt)
    _append_trace(record)
    return record


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _execute(cwd: str, prompt: str, arm: str, budget_s: int,
             max_usd: float, record: dict,
             plugin_dir: str | None = None) -> tuple[str, int | None, float]:
    """跑一次 claude。返回 (status, exit_code, wall_time_s)。

    status ∈ ok / timeout / budget / error
    """
    cmd = build_cmd(cwd, budget_s, max_usd, plugin_dir=plugin_dir)
    env = os.environ.copy()
    env.update(_load_env())
    # 绑卡并行：把卡号写进子进程环境。DCU 上 torch 认 HIP_VISIBLE_DEVICES，
    # 同时设 CUDA_VISIBLE_DEVICES 兼容老代码/第三方库。
    _dev = os.environ.get("OPBENCH_DEVICE", "")
    if _dev:
        env["HIP_VISIBLE_DEVICES"] = _dev
        env["CUDA_VISIBLE_DEVICES"] = _dev

    prefix = _su_prefix(arm)
    if prefix:
        # su - <user> -c "<cmd>"：登录 shell 会丢掉 LAUNCHER 的环境，
        # 而 A 臂连 API 全靠 ENV_FILE 里的 base_url/token。
        # 在子 shell 里显式 source；值不回显。
        #
        # **两条路径都要去尾斜杠**：`_load_env()` 给非 su 路径去掉了 URL 尾斜杠，
        # 但 su 分支是直接在子 shell 里 source env 文件，不走 _load_env()，
        # 所以这里再 rstrip 一次，否则 tosc.env 的 ...4001/ 会拼成 //v1/... → 404。
        _pin = ""
        if _dev:
            _pin = (f"export HIP_VISIBLE_DEVICES={_dev} "
                    f"CUDA_VISIBLE_DEVICES={_dev} && ")
        inner = (
            f"cd {_shq(cwd)} && set -a && . {_shq(ENV_FILE)} && set +a && "
            f"{_pin}"
            f"ANTHROPIC_BASE_URL=${{ANTHROPIC_BASE_URL%/}} && "
            f"{_quote_for_shell(cmd)}"
        )
        argv = prefix + [inner]
        env.pop("ANTHROPIC_AUTH_TOKEN", None)  # 由子 shell 自己 source
    else:
        argv = cmd

    stdout_path = os.path.join(cwd, "_stdout.jsonl")
    stderr_path = os.path.join(cwd, "_stderr.log")
    started = time.monotonic()
    timed_out = False
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(
            argv,
            cwd=None if prefix else cwd,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,  # 超时时能整组杀
        )
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=budget_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # 杀不干净是实测踩过的坑：`su - opbench -c ...` 会在内部**再建一个
            # session**，所以 os.killpg(proc.pid) 只杀得掉 su 本身，
            # 内层的 claude 会逃逸成孤儿进程继续跑（实测有 38 分钟还在跑的）。
            # 这里按子进程树递归杀，并且在 cwd 上做二次确认。
            _kill_tree(proc.pid)
            proc.wait()
            _kill_leftover_in_cwd(cwd)
    wall = round(time.monotonic() - started, 1)

    record["stdout_bytes"] = os.path.getsize(stdout_path)
    record["stderr_bytes"] = os.path.getsize(stderr_path)
    parsed = _parse_claude_json(stdout_path)
    if parsed:
        record["tokens"] = parsed.get("usage")
        record["num_turns"] = parsed.get("num_turns")
        record["permission_denials"] = [
            d.get("tool_name") for d in (parsed.get("permission_denials") or [])
        ]
        record["is_error"] = parsed.get("is_error")
        record["result_head"] = (parsed.get("result") or "")[:400]
        record["model_usage"] = parsed.get("modelUsage")

    if timed_out:
        return "timeout", proc.returncode, wall
    if proc.returncode != 0:
        return "error", proc.returncode, wall
    if parsed is None:
        return "error", proc.returncode, wall

    # 2026-11-25 实测：C 臂三题调用 loop.py 的次数都是 0 —— 它退化成了"挂了 skill 的裸 CC"，
    # 实验最核心的假设（强制迭代）根本没被验证。所以这里硬卡：没有流水就不算数。
    ev = _loop_evidence(cwd)
    record["loop_evidence"] = ev
    if _requires_loop(arm) and not ev:
        return "invalid:no_loop_evidence", proc.returncode, wall
    return "ok", proc.returncode, wall



def _requires_loop(arm: str) -> bool:
    """哪些臂必须留下 loop 迭代证据。

    目前只有 C 臂（toolkit-skills）：它的 loop.py 会把流水写到 <卡片>/runs/loop/*.jsonl。
    D 臂（SkillHub）的 loop 是 skill 形态、产物路径不统一，暂不做硬校验。
    """
    return arm.upper() == "C"


def _loop_evidence(cwd: str) -> list[str]:
    """在 cwd 下找 loop.py 的迭代流水。"""
    import glob
    got = set(glob.glob(os.path.join(cwd, "**", "runs", "loop", "*.jsonl"), recursive=True))
    got.update(glob.glob(os.path.join(cwd, "runs", "loop", "*.jsonl")))
    return sorted(got)


def _kill_tree(pid: int) -> None:
    """递归杀掉 pid 及其所有后代。先 SIGKILL 子进程，再杀自己。"""
    import subprocess as sp

    try:
        out = sp.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout
    except OSError:
        out = ""
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))

    def walk(p: int) -> list[int]:
        got = []
        for c in children.get(p, []):
            got.extend(walk(c))
        got.append(p)
        return got

    for target in walk(pid):
        try:
            os.kill(target, signal.SIGKILL)
        except OSError:
            pass


def _kill_leftover_in_cwd(cwd: str, timeout_s: float = 5.0) -> list[int]:
    """兜底：把 cwd 指向本 attempt 目录、但已脱离父进程的 claude 杀掉。

    为什么要按 cwd 找：`su` 建的新 session 会让进程树断掉，光靠 ppid
    找不到。而每个 attempt 有独立 cwd，这是最可靠的归属依据。
    """
    import subprocess as sp
    import time

    killed: list[int] = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = sp.run(["ps", "-eo", "pid=,args="], capture_output=True,
                         text=True).stdout
        except OSError:
            break
        found = []
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if "claude -p" not in parts[1]:
                continue
            try:
                if os.readlink(f"/proc/{pid}/cwd") == os.path.realpath(cwd):
                    found.append(pid)
            except OSError:
                continue
        if not found:
            break
        for pid in found:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
        time.sleep(0.2)
    return killed


def _shq(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _parse_claude_json(path: str) -> dict | None:
    """`--output-format json` 在最后一行输出一个 result 对象。

    前面可能有告警行（如 `[claude-code:unrecognized_model] {...}`），
    所以从后往前找第一个能解析且带 type=result 的对象。
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def _collect_artifacts(cwd: str) -> list[str]:
    out = []
    for name in ARTIFACT_NAMES:
        p = os.path.join(cwd, name)
        if not os.path.exists(p):
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    out.append(os.path.relpath(os.path.join(root, fn), cwd))
        else:
            out.append(name)
    return sorted(out)


def _write_attempt_meta(cwd: str, record: dict, prompt: str) -> None:
    with open(os.path.join(cwd, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    # prompt 落盘：三臂的 prompt 必须只差"一行资源声明"，事后要能逐字核
    with open(os.path.join(cwd, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)


def _append_trace(record: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    from bench.data import trace as trace_mod

    t = trace_mod.Trace(
        task=record["task"],
        arm=record["arm"],
        started_at=record["started_at"],
        wall_time_s=record["wall_time_s"] or 0.0,
        exit_code=record["exit_code"] if record["exit_code"] is not None else -1,
        sclk_mhz=record["sclk_mhz"],
        libs=_libs_for_trace(record),
        artifacts=record["artifacts"],
        note=record.get("status") or "",
        sclk_end_mhz=record["sclk_end_mhz"],
        tokens=record.get("tokens"),
        budget_s=record["budget_s"],
    )
    trace_mod.append(TRACE_PATH, t)


def _libs_for_trace(record: dict) -> dict | None:
    if not record.get("needs_gpu"):
        return None
    try:
        from bench.hw import dcu

        return dcu.library_versions()
    except Exception:
        return None


# ---------------------------------------------------------------- suite 模式

def load_suite(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data.get("tasks"):
        raise ValueError(f"{path} 里没有 tasks")
    return data


def prepare_suite(suite_path: str, arms: list[str]) -> list[dict]:
    """展开 (task, arm) 队列并建目录。返回队列。"""
    suite = load_suite(suite_path)
    queue = []
    for spec in suite["tasks"]:
        tid = spec["id"]
        budget = int(spec.get("budget") or suite.get("budget_per_task") or 1800)
        td = task_dir(tid)  # 先解析，题不存在就在这一步失败，不要跑到一半
        defn = load_definition(td)
        tid = defn.id  # 与 run_one 一致：目录名可能只是旧称
        for arm in arms:
            queue.append({
                "task": tid,
                "arm": arm.upper(),
                "budget": budget,
                "needs_gpu": needs_gpu(defn),
            })
    return queue


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opbench-run")
    p.add_argument("--task")
    p.add_argument("--arm", choices=["A", "B", "C", "D", "a", "b", "c", "d"])
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    p.add_argument("--suite")
    p.add_argument("--arms", default="A,B,C")
    p.add_argument("--prepare", action="store_true", help="只展开队列，不执行")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    if a.suite:
        arms = [x.strip() for x in a.arms.split(",") if x.strip()]
        queue = prepare_suite(a.suite, arms)
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        if a.prepare:
            return 0
        results = []
        for item in queue:
            print(f"→ {item['task']} / {item['arm']} (budget {item['budget']}s, "
                  f"gpu={item['needs_gpu']})", flush=True)
            try:
                r = run_one(item["task"], item["arm"], item["budget"],
                            max_usd=a.max_usd, dry_run=a.dry_run)
            except Exception as e:
                r = {"task": item["task"], "arm": item["arm"],
                     "status": f"error:{type(e).__name__}", "note": str(e)}
            print(f"   {r.get('status')}  wall={r.get('wall_time_s')}s  "
                  f"artifacts={len(r.get('artifacts') or [])}", flush=True)
            results.append(r)
        return 0

    if not a.task or not a.arm:
        p.error("--task 与 --arm 必填（或用 --suite）")
    r = run_one(a.task, a.arm, a.budget, max_usd=a.max_usd, dry_run=a.dry_run)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
