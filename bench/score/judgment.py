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
SCORE_QUALIFIED = 80
SCORE_THIN = 60
SCORE_HONEST = 20
SCORE_ZERO = 0

# 「定量实测」的判据：报告里出现了带单位的测量值，或误差量（rel_l2/max_abs）。
# 为什么这一档必须有：docs/scoring.md 说满分要「有实测证据」。
# 只贴报错原文 + 命令，说明"我确实调了"，但没说明"我测出了什么"——
# 那是 80 分，不是 100。这条一加，'能跑通'与'跑通并量出数'才分得开。
_QUANT = (
    r"(?<![\w.])\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:ms|us|µs|ns|s|TFLOPS|TF|GB/s|%|x|×)(?![\w.])"
    r"|\b(?:rel_l2|max_abs|max_abs_err|cos|cosine|ratio)\b\s*[=:≈<]?\s*\d"
    r"|[-+]?\d+\.\d+e[-+]\d+"
)


def has_quantitative(text: str) -> bool:
    """报告里有没有定量测量结果（带单位的值 / 误差量 / 科学计数法）。"""
    import re

    return bool(re.search(_QUANT, text or "", re.I))


# 「我自己没打通数值验证」的自承。这类回答有数（cos≈0.10），
# 但那个数恰恰是**失败**的证据。给它满分就是把"我没验成"和"我验成了"
# 混为一谈——实测中 B 臂与 C 臂的差别就在这里。
_UNVERIFIED = (
    "未能在数值上复核", "数值正确性未验证", "对不上", "无法在.*定位",
    "不是已证正确", "未能复现", "cos≈0.1", "rel_max>1",
)


def claims_unverified(text: str) -> str | None:
    """返回命中的自承语句（None 表示没有）。"""
    import re

    for pat in _UNVERIFIED:
        m = re.search(pat, text or "")
        if m:
            return m.group(0)
    return None


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


# expected_verdict → 判定「结论方向对不对」的判据词组。
#
# 这是修掉"答错也得 60 分"的关键：原实现只问"文本里有没有命中 verdict_keywords"，
# 从不比较 expected_verdict。于是只要回答谈到这个话题（命中任何关键词），
# 方向就算"正确"，只有证据量决定 0/60/100——一个答反了的详实报告能拿满分。
#
# 现在把方向判定拆成两步：
#   1. 从回答里**判定它声称的结论**（verdict 极性）
#   2. 与 expected_verdict 比对
# verdict_keywords 的语义随之收紧为「表达该结论的措辞」，不是「话题词」。
VERDICT_ALIASES = {
    "available": ("可用", "可以融合", "支持", "能跑通", "可运行", "available"),
    # 不收单字「没」：它会把「还没测」「没验证」这类诚实声明误判成
    # "结论是不可用"。方向词必须是能独立表达结论的词组。
    "unavailable": ("不可用", "不支持", "不该", "不通", "不能",
                    "unavailable", "not available"),
    "conditional": ("有条件", "受限", "需满足", "前提", "必须显式",
                    "conditional", "仅在"),
}
# expected_verdict 允许写英文枚举（definition.json 的规范）或中文原文
# （早期测试与手写题面用的是这个）。两种都归一成同一个键。
_EXPECTED_NORM = {
    "不可用": "unavailable", "不支持": "unavailable",
    "可用": "available",
    "有条件可用": "conditional", "有条件": "conditional",
}


def _norm_expected(value):
    if not isinstance(value, str):
        return value
    return _EXPECTED_NORM.get(value.strip(), value.strip())
# 题目判为 unavailable 时，出现这些"反方向"措辞即视为答反
OPPOSITE = {
    "unavailable": ("available",),
    "available": ("unavailable",),
    "conditional": ("unavailable", "available"),
}


# 否定语境："不是不可用"、"并非不可用"、"non-unavailable"。
# 不处理它会把「不是"不可用"，但要澄清边界」这类澄清句判成答反。
_NEGATION = ("不是", "并非", "而非", "不等于", "并不是", "不像", "not ", "non-",
             "不能算", "不属于")


def _declared_verdicts(text: str) -> set[str]:
    """回答里出现了哪些结论极性。

    两个坑都要处理：
      1. 方向词天然嵌套：「不可用」包含「可用」。先判否定式，命中就排除肯定式。
      2. **否定语境**：报告会写「不是"不可用"，而是有条件可用」。
         这类句子里的方向词是被否定的，不能算作它的声明。
    做法：逐句扫，句内方向词前若在 12 字符内出现否定词，则该方向词作废。
    """
    import re

    found: set[str] = set()
    # 先按句子切，逐句判定，避免跨句污染
    # 逗号也要切：中文报告常写「并非不可用，而是有条件可用」，
    # 一句话里前半个分句否定、后半个分句肯定。整句扫会把后半句也一起丢掉。
    for sent in re.split(r"[。\n；;.!?，,]", text):
        if not sent.strip():
            continue
        low = sent.lower()
        hits_here: set[str] = set()
        for polarity, words in VERDICT_ALIASES.items():
            for w in words:
                wl = w.lower()
                idx = low.find(wl)
                if idx < 0:
                    continue
                # 句内 12 字符内出现否定词 → 该方向词被否定
                head = low[max(0, idx - 12):idx]
                if any(n in head for n in _NEGATION):
                    continue
                hits_here.add(polarity)
                break
        # 「不可用 / 不支持」优先于「可用」
        if "unavailable" in hits_here:
            hits_here.discard("available")
            if any(w in low for w in ("不可用", "unavailable", "不支持", "not available")):
                hits_here.discard("conditional")
        found |= hits_here
    return found


def score(verdict: str, evidence: list[str], defn) -> tuple[int, str]:
    """返回 (分数, 理由)。defn 提供 expected_verdict / verdict_keywords /
    forbidden_claims / evidence_required。
    """
    text = (verdict or "").strip()
    keywords = _get(defn, "verdict_keywords", []) or []
    forbidden = _get(defn, "forbidden_claims", []) or []
    expected = _norm_expected(_get(defn, "expected_verdict"))
    need, need_kinds = _evidence_need(defn)
    evidence = [e for e in (evidence or []) if str(e).strip()]

    if not text:
        return SCORE_ZERO, "结论为空"

    hits = is_forbidden(text, forbidden)
    if hits:
        return SCORE_ZERO, f"命中被禁止的主张 {hits[:3]}，视为编造证据"

    declared = _declared_verdicts(text)
    matched = [k for k in keywords if k and str(k) in text]
    honest = [m for m in HONEST_MARKERS if m in text]

    # ---- 方向判定：先看显式声明的结论极性 ----
    if expected and expected != "unknown":
        if declared and expected not in declared:
            return (
                SCORE_ZERO,
                f"结论方向错：题面要 {expected}，回答说的是 {sorted(declared)}",
            )
        if not declared and not matched and not honest:
            return SCORE_ZERO, "结论既未命中判据关键词，也未声明不确定（疑似臆断）"
        if not declared and not matched and honest:
            return SCORE_HONEST, f"明确声明「{honest[0]}」，诚实但未给出结论"
    else:
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

    blob = text + "\n" + "\n".join(evidence)

    # 自承"没打通数值验证" → 80，不是 100。它有数，但那是失败的证据。
    # 与下一档的区别：这一档**诚实**（主动说了没验成），所以不是 60 或 0。
    unv = claims_unverified(text)
    if unv:
        return (
            SCORE_QUALIFIED,
            f"结论正确且证据齐全，但自承未完成数值验证（「{unv}」）",
        )

    # 证据种类齐了，但还没有定量结果 → 80，不是 100。
    # 这一步是"三臂全 100"的直接解药：只讲对 + 只贴报错，与真的量出数的，
    # 必须分开计分。注意数字通常在**证据**里而不是结论句里，两者都扫。
    if not has_quantitative(blob):
        return (
            SCORE_QUALIFIED,
            f"结论正确且证据种类齐全（{len(evidence)} 条），但没有定量测量结果",
        )

    return (
        SCORE_FULL,
        f"结论正确（命中 {matched[:2]}）且给出 {len(evidence)} 条证据，含定量结果",
    )
