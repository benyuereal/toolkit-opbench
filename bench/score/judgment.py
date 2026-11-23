"""L3/L5 判定题打分。

不跑 bench，只看结论与证据：
  100  结论正确 + 有实测证据 + 说清机制
   60  结论正确但证据不足
   20  明确说"未验证/不知道"（诚实）
    0  结论错误或编造证据（并计入幻觉）

为什么有 20 分这一档：本项目最大的风险不是不够快，而是**编一个看起来合理的结论**。
历史上出现过三次（in-house 29 条、两个不存在的库、把实现达成值当硬件峰值）。
把"我不知道"给到 20 分，是为了让诚实比臆断得分高——但永远低于有证据的正确答案。
"""

from __future__ import annotations

# 表达"没测/不确定"的标记。命中即视为诚实声明，给 20 分而不是 0 分。
HONEST_MARKERS = (
    "未测定",
    "未验证",
    "未做验证",
    "不确定",
    "不知道",
    "暂无证据",
    "没测",
    "无法确认",
)

SCORE_FULL = 100
SCORE_THIN = 60
SCORE_HONEST = 20
SCORE_ZERO = 0


def _get(defn, key: str, default=None):
    """defn 可能是 dict，也可能是 definition.Definition 这类对象。

    对象优先读 `.raw`（原始 JSON）：属性上可能挂的是同名方法或别的
    表示形式，而评分只该看题面写了什么。没有 raw 再退回 getattr。
    """
    if defn is None:
        return default
    if isinstance(defn, dict):
        return defn.get(key, default)
    raw = getattr(defn, "raw", None)
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    value = getattr(defn, key, default)
    return default if callable(value) else value


def is_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """命中 forbidden_claims 即视为编造（如 L5 题里说"可以融合"）。

    返回命中的片段列表；空列表表示没命中。ASCII 部分不区分大小写。
    """
    if not text or not forbidden:
        return []
    lowered = text.lower()
    hits = []
    for phrase in forbidden:
        if not phrase:
            continue
        p = str(phrase)
        if p.lower() in lowered:
            hits.append(p)
    return hits


def _evidence_need(defn) -> tuple[int, list[str]]:
    """把 evidence_required 归一成 (最少条数, 必需的证据种类)。

    为什么两种形态都收：docs/format.md 定义的是**字符串数组**
    （["error_output", "command"]），而早期实现按整数条数处理。题面一旦
    用正式格式，`int(list)` 会直接 TypeError。这里以 format.md 为准，
    同时兼容整数写法：
      * int n        → 只要 n 条证据
      * list[str]    → 条数 = len(list)，且每条种类都要在证据里出现
    """
    req = _get(defn, "evidence_required", 1)
    if req is None:
        return 1, []
    if isinstance(req, (int, float)) and not isinstance(req, bool):
        return max(1, int(req)), []
    if isinstance(req, str):
        return 1, [req]
    if isinstance(req, (list, tuple)):
        kinds = [str(k) for k in req if str(k).strip()]
        return max(1, len(kinds)), kinds
    return 1, []


# 证据种类的同义词。要求报告里出现 ASCII 词 "error_output" 是不合理的——
# 中文报告会写「报错原文」。这里只放**确定指向该种类**的说法，
# 不用泛词（如「测试」），否则任何报告都能凑满。
KIND_SYNONYMS = {
    "error_output": ("error_output", "error", "traceback", "exception", "runtimeerror",
                     "报错", "错误", "异常"),
    "command": ("command", "cmd", "python3 ", "bash ", "sh -c", "命令", "复跑"),
    "import": ("import", "导入"),
    "call": ("call", "invoke", "调用", "调了", "实测调用"),
    "log": ("log", "日志"),
    "results": ("results/", "结果文件", "results.json"),
}


def _missing_kinds(kinds: list[str], evidence: list[str]) -> list[str]:
    """required 里的每种证据，必须在 evidence 文本里出现（不区分大小写）。

    只数条数是不够的：交两行"我测了"也能凑够 2 条。种类绑定是为了逼
    证据落在题面要求的维度上（报错原文、复跑命令、调用路径……）。
    匹配时认同义词（见 KIND_SYNONYMS），因为报告是中文的。
    """
    if not kinds:
        return []
    blob = "\n".join(evidence).lower()
    missing = []
    for k in kinds:
        alts = KIND_SYNONYMS.get(k, (k,))
        if not any(a.lower() in blob for a in alts):
            missing.append(k)
    return missing


def score(verdict: str, evidence: list[str], defn) -> tuple[int, str]:
    """返回 (分数, 理由)。defn 提供 expected_verdict / verdict_keywords /
    forbidden_claims / evidence_required。
    """
    text = (verdict or "").strip()
    keywords = _get(defn, "verdict_keywords", []) or []
    forbidden = _get(defn, "forbidden_claims", []) or []
    need, need_kinds = _evidence_need(defn)
    evidence = [e for e in (evidence or []) if str(e).strip()]

    if not text:
        return SCORE_ZERO, "结论为空"

    hits = is_forbidden(text, forbidden)
    if hits:
        return SCORE_ZERO, f"命中被禁止的主张 {hits[:3]}，视为编造证据"

    matched = [k for k in keywords if k and str(k) in text]
    honest = [m for m in HONEST_MARKERS if m in text]

    if not matched and not honest:
        return SCORE_ZERO, "结论既未命中判据关键词，也未声明不确定（疑似臆断）"

    if not matched and honest:
        return SCORE_HONEST, f"明确声明「{honest[0]}」，诚实但未给出结论"

    missing = _missing_kinds(need_kinds, evidence)
    if len(evidence) < need or missing:
        why = f"结论方向正确（命中 {matched[:2]}），但证据 {len(evidence)} 条少于要求的 {need} 条"
        if missing:
            why = (
                f"结论方向正确（命中 {matched[:2]}），但缺少要求的证据种类 {missing}"
            )
        return SCORE_THIN, why

    return (
        SCORE_FULL,
        f"结论正确（命中 {matched[:2]}）且给出 {len(evidence)} 条证据",
    )
