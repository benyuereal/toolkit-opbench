"""单臂单题执行器：隔离 / 环境注入 / 超时 / 产物收集。

执行契约（与 protocol.md 一致）：
  1. 建 runs/<arm>/<task>/ 作为 cwd，chmod 777（A 臂要以 opeval 用户写）
  2. 注入上下文（A 无 / B 只给 libraries-registry.yaml / C 给 skill 仓库）
  3. 预热时钟到工作档，记录 sclk
  4. timeout <budget> claude -p "$(cat task.md)" --output-format json
  5. 收集 report.md / scripts/ / results/
  6. 写 trace.jsonl：wall time、tokens、sclk、exit code
超时即判失败，不续跑——补跑会破坏固定预算的可比性。
"""
import argparse


def build_cmd(task_dir: str, budget_s: int) -> list[str]:
    raise NotImplementedError


def prepare_cwd(task_id: str, arm: str) -> str:
    raise NotImplementedError


def run_one(task_id: str, arm: str, budget_s: int) -> dict:
    raise NotImplementedError


def main(argv=None):
    p = argparse.ArgumentParser(prog="opbench-run")
    p.add_argument("--task", required=True)
    p.add_argument("--arm", required=True, choices=["A", "B", "C"])
    p.add_argument("--budget", type=int, default=1800)
    a = p.parse_args(argv)
    return run_one(a.task, a.arm, a.budget)


if __name__ == "__main__":
    main()
