"""体检：一条命令回答"这台机器现在能不能测"。

任一项红，都必须说清怎么修——不是只报错。
"""
CHECKS = [
    ("devices", "8 张卡可见"),
    ("event_timing", "torch.cuda.Event 可计时"),
    ("l2_size", "L2 = 8 MB"),
    ("sclk", "当前 sclk（空闲 600 / 工作 1150 MHz）"),
    ("clock_lock", "能否锁频（hy-smi --save/--load）"),
    ("libs", "lightop / aiter / deepgemm / torch / DTK 版本"),
    ("gpu_free", "另一会话是否占卡（HCU% <= 20）"),
    ("constants", "constants.yaml 三类峰值齐全"),
]


def run() -> dict:
    """逐项跑，返回 {name: {ok, value, fix_hint}}。"""
    raise NotImplementedError


def render(table: dict) -> str:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(render(run()))
