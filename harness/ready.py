"""Bench 自检：一条命令回答"这套 bench 现在能不能用"。

    python3 -m harness.ready

## 检查项（每项都能独立定位问题）

  task_schema     每道题的定义能加载、过校验
  task_complete   kernel 题六件套齐全；judgment 题三件套齐全
  answer_isolated 答案文件（expect.md / definition.json）对 A 臂用户不可读
  baseline        kernel 题有基线，且基线数值自洽
  correctness     reference 自检通过（reference vs reference 恒等）
  judge_scoring   judgment 题的评分器能给出区分（合格100 / 未验证20）
  tooling         verify/score/collect/measure 的入口都在
  gpu             GPU 可用、时钟一致、有没有空卡

## 与 doctor 的分工

`doctor` 回答"**这台机器**现在能不能测"（GPU/库/时钟）。
`ready` 回答"**这套 bench** 现在能不能用"（题目/基线/工具链）。
两者都要绿才能开跑。
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KERNEL_FILES = ("task.md", "definition.json", "workload.jsonl",
                "reference.py", "check.py", "expect.md")
JUDGMENT_FILES = ("task.md", "definition.json", "expect.md")


def _ok(v, hint=""):
    return {"ok": True, "level": "ok", "value": v, "fix_hint": hint}


def _bad(v, hint):
    return {"ok": False, "level": "fail", "value": v, "fix_hint": hint}


def _warn(v, hint):
    return {"ok": False, "level": "warn", "value": v, "fix_hint": hint}


def check_task_schema() -> dict:
    from bench.data import definition as defn_mod

    bad = []
    n = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", "definition.json"))):
        n += 1
        try:
            defn_mod.load(p)
        except Exception as e:
            bad.append(f"{os.path.relpath(p, ROOT)}: {e}")
    if bad:
        return _bad(f"{len(bad)}/{n} 道题定义有问题", "；".join(bad[:3]))
    return _ok(f"{n} 道题定义均通过校验")


def check_task_complete() -> dict:
    from bench.data import definition as defn_mod

    incomplete = []
    for p in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", "definition.json"))):
        td = os.path.dirname(p)
        if os.path.basename(td).startswith("_"):
            continue
        try:
            d = defn_mod.load(p)
        except Exception:
            continue
        need = KERNEL_FILES if d.kind == "kernel" else JUDGMENT_FILES
        miss = [f for f in need if not os.path.isfile(os.path.join(td, f))]
        if miss:
            incomplete.append(f"{d.id}: 缺 {miss}")
    if incomplete:
        return _bad(f"{len(incomplete)} 道题不完整", "；".join(incomplete[:3]))
    return _ok("所有题的文件齐全")


def check_answer_isolated() -> dict:
    """答案文件必须对 others 不可读（A 臂以 opbench 用户跑）。

    三类都算答案文件：
      * expect.md       —— 结论与评分细则
      * definition.json —— 含 baseline_impl 等口径
      * reference.py    —— **逐行可执行的正确实现**，读到它比读 expect.md
                          更省事：直接抄语义，连猜都不用猜。

    为什么必须按"others 位"（0o004）查而不是按没说 644 查：跑臂用的是
    独立用户（opb_a/opb_c/opb_d），只要 others 有读权限就读得到。
    reference.py 之前不在这张名单里，实测对三个臂用户全部可读——题面
    白写，答案敞开。
    """
    leaks = []
    for pat, want in (
        ("expect.md", "chmod 600"),
        ("definition.json", "chmod 640"),
        ("reference.py", "chmod 640"),
    ):
        for p in glob.glob(os.path.join(ROOT, "tasks", "*", pat)):
            if os.stat(p).st_mode & 0o004:
                leaks.append((os.path.relpath(p, ROOT), want))
    if leaks:
        return _bad(
            f"{len(leaks)} 个答案文件对 others 可读",
            "按类型收紧：expect.md → 600，definition.json / reference.py → 640。"
            "例如：" + "；".join(f"{w} {p}" for p, w in leaks[:2]),
        )
    return _ok("答案文件均隔离（expect/definition/reference）")


def check_baseline() -> dict:
    from bench.data import definition as defn_mod

    missing, wrong = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", "definition.json"))):
        try:
            d = defn_mod.load(p)
        except Exception:
            continue
        if d.kind != "kernel" or os.path.basename(os.path.dirname(p)).startswith("_"):
            continue
        bp = os.path.join(ROOT, "baselines", f"{d.id}.json")
        if not os.path.isfile(bp):
            missing.append(d.id)
            continue
        b = json.load(open(bp, encoding="utf-8"))
        if not (b.get("t_b_ms", 0) > 0 and b.get("t_sol_ms", 0) > 0):
            wrong.append(f"{d.id}: t_b/t_sol 非正")
        elif b.get("sclk_mhz") is None:
            wrong.append(f"{d.id}: 没记 sclk")
    if missing or wrong:
        return _bad(f"缺 {len(missing)} 道基线；异常 {len(wrong)} 道",
                    f"缺:{missing} 异常:{wrong}；跑 python3 -m harness.baseline --task <id>")
    return _ok("所有 kernel 题都有自洽基线")


def check_correctness() -> dict:
    """reference 自检：reference 与自身对比必须恒等。"""
    import importlib.util

    from bench.check import correctness as C
    from bench.data import definition as defn_mod
    from bench.data import workload as wl_mod

    failed = []
    n = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", "definition.json"))):
        td = os.path.dirname(p)
        try:
            d = defn_mod.load(p)
        except Exception:
            continue
        if d.kind != "kernel" or not os.path.isfile(os.path.join(td, "reference.py")):
            continue
        n += 1
        try:
            spec = importlib.util.spec_from_file_location(
                f"ref_{d.id}", os.path.join(td, "reference.py"))
            ref = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ref)
            ws = wl_mod.load(os.path.join(td, "workload.jsonl"), d)
            r = C.check_all(d, ws, lambda t: ref.arun(*[t[k] for k in d.inputs]), ref)
            if not r.ok:
                failed.append(f"{d.id}: {(r.note or '')[:60]}")
        except Exception as e:
            failed.append(f"{d.id}: {type(e).__name__}: {str(e)[:60]}")
    if failed:
        return _bad(f"{len(failed)}/{n} 道题 reference 自检失败",
                    "；".join(failed[:3]))
    return _ok(f"{n} 道 kernel 题 reference 自检通过")


def check_judge_scoring() -> dict:
    from bench.score import judgment as J

    problems = []
    for p in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", "definition.json"))):
        from bench.data import definition as defn_mod
        try:
            d = defn_mod.load(p)
        except Exception:
            continue
        if d.kind != "judgment":
            continue
        kw = d.verdict_keywords
        good = kw[0] if kw else ""
        s_ok = J.score(f"{good}。python3 x.py 报错 RuntimeError: boom rel_l2 1e-3",
                       ["error_output: RuntimeError", "command: python3 x.py",
                        "rel_l2 1e-3", "import aiter 调用"], d)[0]
        s_naive = J.score("未验证。", [], d)[0]
        if s_ok != 100 or s_naive != 20:
            problems.append(f"{d.id}: 合格={s_ok} 未验证={s_naive}")
    if problems:
        return _bad(f"{len(problems)} 道 judgment 题打分异常", "；".join(problems[:3]))
    return _ok("judgment 打分器区分正常（100/20）")


def check_tooling() -> dict:
    mods = ["bench.check.integrity", "bench.score.ratio", "harness.collect",
            "harness.measure", "harness.baseline", "harness.doctor"]
    missing = []
    for m in mods:
        r = subprocess.run([sys.executable, "-m", m, "--help"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            missing.append(m)
    if missing:
        return _bad(f"{len(missing)} 个工具入口不可用", f"缺: {missing}")
    return _ok(f"{len(mods)} 个工具入口均可用")


def check_arm_writable() -> dict:
    """每个臂的用户，是否真能往它将来的工作目录里写文件。

    ## 为什么要单独查这一条

    2026-11-26 那天，三题并行跑了 9 分钟，两题的 A 臂零产出。原因不是
    agent 不干活，是 `scripts/` 被建成 `root:root 755`，它 `write
    scripts/x.py` 报 Permission denied，然后反复换路径重试到预算耗尽。

    这种失败**不会报错、不会留日志**，表现就是"这个臂没产出"——看起来
    像模型不行。它能安静地毁掉一整轮对照实验，所以必须在上bench 之前
    用一个真实的写操作验一遍，而不是猜权限位。

    查法：以该臂用户的身份，往一个临时目录里 touch 一个文件。用 `su`
    而不是看 `os.access`——后者查的是当前进程（root）的权限，root
    什么都能写，查了等于没查。
    """
    import tempfile

    from harness.launch import _arm_module

    arms = ["A", "C", "D"]
    broken, skipped = [], []
    for arm in arms:
        try:
            mod = _arm_module(arm)
        except Exception as e:
            skipped.append(f"{arm}(取不到臂模块: {e})")
            continue
        user = getattr(mod, "USER", None)
        if not user:
            continue
        d = tempfile.mkdtemp(prefix=f"opb-w-{arm}-")
        try:
            os.chmod(d, 0o777)
            probe = os.path.join(d, "scripts")
            os.makedirs(probe, exist_ok=True)
            os.chmod(probe, 0o777)
            target = os.path.join(probe, ".write-probe")
            r = subprocess.run(
                ["su", "-", user, "-c", f"touch {target}"],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode != 0:
                broken.append(f"{arm}({user}): {(r.stderr or '').strip()[:80]}")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    if broken:
        return _bad(
            f"{len(broken)} 个臂的用户写不进工作目录",
            "；".join(broken) + "。这类失败不抛异常，只会表现为『该臂没产出』",
        )
    msg = f"{len(arms) - len(skipped)} 个臂的用户均可写"
    if skipped:
        msg += f"（跳过 {skipped}）"
    return _ok(msg)


def check_gpu() -> dict:
    try:
        from bench.hw import dcu

        devs = dcu.device_count()
        if devs == 0:
            return _bad("0 张卡", "先修 GPU 环境（见 harness.doctor）")
        free = dcu.free_devices()
        ok, levels = (lambda: (len({lv for lv, _ in dcu._sclk_levels()}) <= 1,
                               sorted({lv for lv, _ in dcu._sclk_levels()})))()
        if not ok:
            return _warn(f"卡时钟档位不一致 {levels}",
                         "跨卡并行前必须统一档位，否则跨卡耗时不可比")
        if not free:
            return _warn("没有空闲卡", "claude-work 可能在用卡，等它空下来")
        return _ok(f"{devs} 张卡，时钟一致（档位 {levels}），空闲 {free}")
    except Exception as e:
        return _bad(f"{type(e).__name__}: {e}", "见 harness.doctor")


CHECKS = {
    "task_schema": check_task_schema,
    "task_complete": check_task_complete,
    "answer_isolated": check_answer_isolated,
    "baseline": check_baseline,
    "correctness": check_correctness,
    "judge_scoring": check_judge_scoring,
    "tooling": check_tooling,
    "arm_writable": check_arm_writable,
    "gpu": check_gpu,
}


def run() -> dict:
    out = {}
    for name, fn in CHECKS.items():
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = _bad(f"{type(e).__name__}: {e}", "这是自检自身的异常，先报给维护者")
    return out


_MARK = {"ok": "✓", "warn": "!", "fail": "✗"}


def render(table: dict) -> str:
    lines = ["opbench ready — 这套 bench 现在能不能用", "=" * 72]
    fails = warns = 0
    for name in CHECKS:
        item = table.get(name)
        if not item:
            continue
        lvl = item["level"]
        fails += lvl == "fail"
        warns += lvl == "warn"
        lines.append(f"[{_MARK[lvl]}] {name:<16} {item['value']}")
        if item.get("fix_hint") and lvl != "ok":
            lines.append(f"      → {item['fix_hint']}")
    lines.append("-" * 72)
    verdict = ("可以开跑" if fails == 0 and warns == 0
               else "可以跑，但有 ! 项要留意" if fails == 0 else "不能跑：先修 ✗")
    lines.append(f"{len(table)} 项：{len(table)-fails-warns} ok / {warns} warn / {fails} fail ⇒ {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    t = run()
    if "--json" in (argv or sys.argv[1:]):
        print(json.dumps(t, ensure_ascii=False, indent=2))
    else:
        print(render(t))
    return 1 if any(i["level"] == "fail" for i in t.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
