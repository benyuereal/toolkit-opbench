"""看每条臂**实际在做什么**——从它自己的 claude session 里读，不是猜文件时间。

    python3 -m harness.multi.watch --task norm-act-swiglu-add-rmsnorm-l1
    python3 -m harness.multi.watch --task ... --tail 12       # 每个臂最后 12 个动作
    python3 -m harness.multi.watch --task ... --once          # 不刷屏，只打一次
    python3 -m harness.multi.watch --task ... --grep loop     # 只看含关键词的动作

## 为什么读 session 而不是只看产物

产物只能回答"有没有东西落地"，回答不了"它现在在想什么、走到哪一步了"。
实测中这两种状态在文件上完全一样：

  * 正在读 skill 卡、还没动手（产物为空，但工作正常）
  * 卡在一个错误的 cd 上打转（产物也为空，但已经浪费 20 分钟）

claude 每次 -p 运行都会把完整会话写进
`<home>/.claude/projects/<cwd-slug>/<session>.jsonl`。读它能看到：
每一条 `tool_use`（跑了什么命令）、每一条 assistant 文本（它的判断）、
以及工具返回（命令是否失败）。这是判断"走没走对入口"的唯一硬证据。

## 安全

只读，不写任何东西。session 文件属于各臂用户，但我们在 root 下跑，
读得到且不会改变任何状态。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pwd
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOLATED_ROOT = "/tmp/opbench-run"


def _slug(cwd: str) -> str:
    """claude 把 cwd 路径里的 / 换成 -，作为 projects 下的目录名。"""
    return cwd.replace("/", "-")


def arm_cwd(task: str, arm: str) -> str | None:
    """最新的 attempt 目录。"""
    pat = os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}", task, "attempt-*")
    dirs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    return dirs[-1] if dirs else None


def arm_user(arm: str) -> str | None:
    from harness.run import _arm_module

    return getattr(_arm_module(arm), "USER", None)


def session_file(task: str, arm: str) -> str | None:
    cwd = arm_cwd(task, arm)
    if not cwd:
        return None
    user = arm_user(arm)
    if not user:
        return None
    try:
        home = pwd.getpwnam(user).pw_dir
    except KeyError:
        return None
    pat = os.path.join(home, ".claude", "projects", _slug(cwd), "*.jsonl")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    return files[-1] if files else None


def read_events(path: str) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def parse_actions(events: list[dict]) -> list[dict]:
    """把 session 记录压成 [{kind, label, detail, ok}]。

    kind: say（它的判断） / tool（调了什么） / fail（工具返回失败）
    """
    actions: list[dict] = []
    for r in events:
        t = r.get("type")
        msg = r.get("message") or {}
        if t == "assistant":
            for c in msg.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text" and (c.get("text") or "").strip():
                    actions.append({
                        "kind": "say",
                        "label": c["text"].strip().replace("\n", " ")[:160],
                    })
                elif c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    cmd = inp.get("command") or inp.get("file_path") or \
                        inp.get("pattern") or inp.get("skill") or ""
                    actions.append({
                        "kind": "tool",
                        "label": f"{c.get('name')}: {str(cmd)[:200]}",
                    })
        elif t == "user":
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict) or c.get("type") != "tool_result":
                    continue
                body = c.get("content")
                if isinstance(body, list):
                    body = " ".join(x.get("text", "") for x in body if isinstance(x, dict))
                body = str(body or "")
                low = body.lower()
                # 只把明确的失败标出来，避免把正常 stderr 也算失败
                bad = ("no such file" in low or "command not found" in low
                       or "traceback" in low or "error:" in low[:200])
                if bad or c.get("is_error"):
                    actions.append({"kind": "fail",
                                    "label": body.strip()[:160].replace("\n", " ")})
    return actions


def status_line(task: str, arm: str) -> dict:
    """一条臂的当前状态摘要。"""
    cwd = arm_cwd(task, arm)
    sf = session_file(task, arm)
    info = {
        "arm": arm,
        "cwd": cwd,
        "session": sf,
        "alive": False,
        "n_actions": 0,
        "last": None,
        "has_bench": False,
        "has_report": False,
        "idle_s": None,
    }
    if cwd:
        info["has_bench"] = os.path.isfile(os.path.join(cwd, "results", "bench_std.json"))
        info["has_report"] = os.path.isfile(os.path.join(cwd, "report.md"))
    if sf:
        info["n_actions"] = len(read_events(sf))
        info["idle_s"] = time.time() - os.path.getmtime(sf)
    user = arm_user(arm)
    if user:
        try:
            p = subprocess.run(["pgrep", "-u", user, "-f", "opbench-node/bin/claude"],
                               capture_output=True, timeout=5)
            info["alive"] = bool(p.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return info


def render_task(task: str, arms: list[str], tail: int = 6,
                grep: str | None = None) -> str:
    lines = [f"═══ {task} ═══"]
    for arm in arms:
        s = status_line(task, arm)
        state = "运行中" if s["alive"] else ("有产物" if (s["has_bench"] or s["has_report"]) else "已停止")
        idle = f"{s['idle_s']/60:.0f}min 前更新" if s["idle_s"] else "无 session"
        lines.append(
            f"\n▸ 臂 {arm}  [{state}]  动作 {s['n_actions']} 条  {idle}  "
            f"bench={'Y' if s['has_bench'] else '-'} report={'Y' if s['has_report'] else '-'}"
        )
        if not s["session"]:
            lines.append("   (没有 session 文件：还没启动，或 cwd 变了)")
            continue
        acts = parse_actions(read_events(s["session"]))
        if grep:
            acts = [a for a in acts if grep.lower() in a["label"].lower()]
        for a in acts[-tail:]:
            mark = {"say": "🤖", "tool": "  ⚙", "fail": "  ✗"}[a["kind"]]
            lines.append(f"   {mark} {a['label']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="opbench-watch",
        description="看每条臂实际在做什么（读各自的 claude session）",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--arms", default="A,C,D")
    p.add_argument("--tail", type=int, default=6, help="每条臂显示最后几个动作")
    p.add_argument("--grep", help="只看包含该关键词的动作")
    p.add_argument("--once", action="store_true", help="只打一次，不循环刷新")
    p.add_argument("--interval", type=int, default=30, help="刷新秒数")
    a = p.parse_args(argv)

    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    if a.once:
        print(render_task(a.task, arms, a.tail, a.grep))
        return 0

    try:
        while True:
            os.system("clear")
            print(render_task(a.task, arms, a.tail, a.grep))
            print(f"\n(Ctrl-C 退出；每 {a.interval}s 刷新)")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
