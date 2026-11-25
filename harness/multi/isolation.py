"""臂间隔离：每个臂一个 Unix 用户 + 一个工作根目录，彼此不可见。

## 为什么必须到"用户"这一级

只把目录设成 700 不够——历史上三个臂都用 `opbench` 用户跑，结果
**实测 C 臂读了 A 臂的 `report.md`**。同一个 uid 下，文件权限形同虚设。

所以现在是**一臂一用户**：

    A → opb_a   /tmp/opbench-run/arm-A/...
    C → opb_c   /tmp/opbench-run/arm-C/...
    D → opb_d   /tmp/opbench-run/arm-D/...

每个 `arm-X` 根目录 `chown` 给该用户 + `chmod 700`，于是：

* 别的臂（另一个 uid）进不来，看不到对方的 `report.md`、`scripts/`、日志；
* 臂自己也**出不去**到仓库：cwd 在 `/tmp` 而不是仓库里，`..` 摸不到
  `tasks/<id>/expect.md`；
* 唯一的共享只读源（skill 目录）是**拷贝**进 cwd 的，不是软链——
  软链会让 agent 顺路径摸回源仓库，C 臂就变成了"能读全部 skill 源"。

## 为什么 A 臂在仓库外

A 臂是裸 CC，它唯一的合法输入就是 prompt。cwd 放进仓库后，它可以用
`..` 找到 `tasks/`、`baselines/`、`harness/`，甚至直接读 `reference.py`。
放 `/tmp` 是让"读不到"成为文件系统的性质，而不是靠 agent 自觉。

## 已知的历史污染

`runs/arm-{A,B,C,D}` 是早期（未隔离时）的落点，权限有 777、属主是 root。
那些目录**不能**参与新的对照——里面的产物可能是在能互读的条件下产生的。
`scan_legacy()` 把它们列出来，`quarantine_legacy()` 收紧权限。

## 必需的 uid

`ensure_users()` 检查 opb_a/opb_c/opb_d 是否存在。缺一个就直接拒绝开跑：
少了独立用户，那条臂会退化回"和别的臂同 uid"，隔离悄悄失效——这种
失败不会报错，只会让 Δ 变脏。
"""
from __future__ import annotations

import os
import pwd
import shutil
import stat
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOLATED_ROOT = "/tmp/opbench-run"
LEGACY_ROOT = os.path.join(ROOT, "runs")


def arm_user(arm: str) -> str | None:
    """从臂模块读 USER。没有独立用户的臂返回 None（不该出现在多臂对照里）。"""
    from harness.run import _arm_module

    return getattr(_arm_module(arm), "USER", None)


def ensure_users(arms: list[str]) -> list[str]:
    """确认每条臂都有独立用户。返回问题清单。

    没有独立用户的臂不能参与"互不可见"的对照——它和别的臂同 uid，
    `chmod 700` 对它无效。这个检查要在建目录之前做。
    """
    problems: list[str] = []
    seen: dict[str, list[str]] = {}
    for arm in arms:
        user = arm_user(arm)
        if not user:
            problems.append(
                f"臂 {arm} 没有独立 Unix 用户（模块里没定义 USER）。\n"
                f"    同一个 uid 下 700 不生效，该臂能看到别的臂的产物。\n"
                f"    修法：在 harness/arms/{arm.lower()}.py 里设 USER='opb_{arm.lower()}'，"
                f"并 useradd 该用户"
            )
            continue
        seen.setdefault(user, []).append(arm)
        try:
            pwd.getpwnam(user)
        except KeyError:
            problems.append(
                f"臂 {arm} 的用户 {user!r} 不存在。\n"
                f"    修法：useradd -m {user}（需要 root）"
            )
    for user, owners in seen.items():
        if len(owners) > 1:
            problems.append(
                f"臂 {owners} 共用用户 {user!r}，彼此可互读。一臂一用户是硬要求。"
            )
    return problems


def claim_tree(arm: str, path: str) -> bool:
    """把 path 整棵树交给该臂的用户并设 700。返回是否成功。"""
    if os.geteuid() != 0:
        return False
    user = arm_user(arm)
    if not user:
        return False
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return False
    ok = True
    for dirpath, dirnames, filenames in os.walk(path):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            try:
                os.chown(p, pw.pw_uid, pw.pw_gid, follow_symlinks=False)
            except OSError:
                ok = False
    for target in (path, *[os.path.join(path, d) for d in os.listdir(path)]):
        try:
            os.chown(target, pw.pw_uid, pw.pw_gid)
            os.chmod(target, 0o700)
        except OSError:
            ok = False
    return ok


def arm_root(arm: str) -> str:
    return os.path.join(ISOLATED_ROOT, f"arm-{arm.upper()}")


def verify_isolation(arms: list[str]) -> list[str]:
    """用**切换用户实测**的方式验证互不可见。返回问题清单。

    不靠"看权限位猜"：直接 su 到每个臂用户，去 stat 别的臂的根目录。
    权限位算错（比如 ACL、父目录某级是 755）只有实测才看得出来。
    """
    if os.geteuid() != 0:
        return ["非 root 无法实测隔离：su 到臂用户会失败"]
    problems: list[str] = []
    for arm in arms:
        user = arm_user(arm)
        if not user:
            continue
        for other in arms:
            if other == arm:
                continue
            probe = arm_root(other)
            if not os.path.isdir(probe):
                continue
            r = subprocess.run(
                ["su", "-", user, "-c", f"ls {probe} 2>/dev/null"],
                capture_output=True, text=True, timeout=15,
            )
            if r.stdout.strip():
                problems.append(
                    f"臂 {arm}（{user}）能列出臂 {other} 的目录 {probe}："
                    f"{r.stdout.strip()[:80]}"
                )
    return problems


def scan_legacy() -> list[dict]:
    """列出 runs/arm-* 里的旧产物。这些不是隔离条件下产生的，不能进新对照。"""
    out: list[dict] = []
    if not os.path.isdir(LEGACY_ROOT):
        return out
    for name in sorted(os.listdir(LEGACY_ROOT)):
        p = os.path.join(LEGACY_ROOT, name)
        if not os.path.isdir(p) or not name.startswith("arm-"):
            continue
        st = os.stat(p)
        world_readable = bool(st.st_mode & stat.S_IROTH)
        out.append({
            "path": os.path.relpath(p, ROOT),
            "arm": name.split("-", 1)[1].split(".")[0],
            "mode": oct(st.st_mode & 0o777),
            "owner": pwd.getpwuid(st.st_uid).pw_name,
            "world_readable": world_readable,
        })
    return out


def quarantine_legacy() -> list[str]:
    """把旧产物树收到 root 700。返回处理过的路径。"""
    done: list[str] = []
    if os.geteuid() != 0:
        return done
    for rec in scan_legacy():
        p = os.path.join(ROOT, rec["path"])
        try:
            subprocess.run(["chmod", "-R", "go-rwx", p], check=False, timeout=120)
            done.append(rec["path"])
        except (OSError, subprocess.SubprocessError):
            pass
    return done
