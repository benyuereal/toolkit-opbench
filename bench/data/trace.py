"""trace.jsonl：每次运行一条记录，供审计与复跑。

必须有：task/arm/时间戳/wall_time/exit_code/sclk/库版本/产物路径。
少了 sclk 与库版本，历史数据就不可比。
"""
from dataclasses import dataclass, asdict, fields
import json

REQUIRED = ("task", "arm", "started_at", "wall_time_s", "exit_code")


@dataclass
class Trace:
    task: str
    arm: str
    started_at: str
    wall_time_s: float
    exit_code: int
    sclk_mhz: int | None = None
    libs: dict | None = None
    artifacts: list[str] | None = None
    note: str = ""
    # 以下为执行器回填，不影响可比性但影响可审计性
    sclk_end_mhz: int | None = None
    tokens: dict | None = None
    attempt: int = 1
    budget_s: int | None = None


_FIELDS = {f.name for f in fields(Trace)}


def append(path: str, t: Trace) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def from_dict(d: dict) -> Trace:
    """宽松读：未知字段进 note 之前先丢掉，不让旧 trace 打断加载。"""
    if not isinstance(d, dict):
        raise ValueError("trace 记录必须是 JSON 对象")
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"trace 记录缺必填字段 {missing}")
    known = {k: v for k, v in d.items() if k in _FIELDS}
    return Trace(**known)


def load(path: str) -> list[Trace]:
    out: list[Trace] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno} 无效：{e}") from e
    return out
