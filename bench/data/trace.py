"""trace.jsonl：每次运行一条记录，供审计与复跑。

必须有：task/arm/时间戳/wall_time/exit_code/sclk/库版本/产物路径。
少了 sclk 与库版本，历史数据就不可比。
"""
from dataclasses import dataclass, asdict
import json


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


def append(path: str, t: Trace) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def load(path: str) -> list[Trace]:
    raise NotImplementedError
