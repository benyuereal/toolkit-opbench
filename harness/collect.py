"""汇总：runs/ → reports/。

产出三臂对比表与 Δ：
  Δ_eff    有效结论率差
  Δ_halluc 幻觉率差（期望 <= 0）
  Δ_calib  校准误差
胜负线跑前定死，这里只做计算与渲染。
"""


def collect(runs_dir: str) -> dict:
    raise NotImplementedError


def render_report(table: dict, template: str | None = None) -> str:
    raise NotImplementedError
