"""
主导格式一致性检测：在"格式高度统一"的列里，标记格式形态偏离主导形态的值。

动机（movies）：脏数据把单一格式改成了另一种表示，且 Stage 1 的 LLM 规则归纳是在
脏数据上做的，常把异常形态当成"合法变体"纳入 regex（如 Year `^\\d{4} \\d{4} \\d{4}$`、
Id slug、Duration 时分制），导致规则放行。直接以"列内主导形态"为基准更稳健。

形态抽象：字母段 -> 'A'，数字段 -> '9'，其余字符保留（如空格/逗号/点/斜杠）。
  '96 min'        -> '9 A'
  '1 hr. 39 min.' -> '9 A. 9 A.'
  'tt1253864'     -> 'A9'
  'immortals_2011'-> 'A_9'

双门控（关键，降误报、自动跳过自由文本/合法多形态列）：
  1. 主导形态占比 >= dom_min：列足够"格式统一"才启用（自由文本列形态分散，被跳过）。
  2. 次高形态占比 <= sec_max：排除"存在第二种合法形态"的列（如 City/src 长短不一），
     这类列偏离主导未必是错误。
经 movies/rayyan/flights/beers/hospital/billionaire 离线验证：dom_min=0.97,sec_max=0.02
时整体精度约 0.98，flights `src`、hospital `City` 等合法多形态列被自动排除。
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from stage_1.profiling import is_blank

_ALPHA = re.compile(r"[A-Za-z]+")
_DIGIT = re.compile(r"\d+")


def value_shape(value: str) -> str:
    """把值抽象为粗格式形态（字母段->A，数字段->9，其余保留）。"""
    s = _ALPHA.sub("A", str(value))
    s = _DIGIT.sub("9", s)
    return s


def detect_format_outliers(
    series: pd.Series,
    column: str,
    *,
    min_rows: int = 30,
    dom_min: float = 0.97,
    sec_max: float = 0.02,
) -> list[dict]:
    """扫描单列，返回"格式形态偏离主导形态"的候选错误（默认交 Stage 3 复核）。"""
    vals = [(rid, str(v)) for rid, v in series.items() if not is_blank(v)]
    n = len(vals)
    if n < min_rows:
        return []
    cnt = Counter(value_shape(v) for _, v in vals)
    ordered = cnt.most_common()
    dom_shape, dom_n = ordered[0]
    dom_ratio = dom_n / n
    sec_ratio = (ordered[1][1] / n) if len(ordered) > 1 else 0.0
    if dom_ratio < dom_min or sec_ratio > sec_max:
        return []

    errors: list[dict] = []
    for rid, v in vals:
        if value_shape(v) == dom_shape:
            continue
        errors.append({
            "row_id": rid,
            "column": column,
            "value": v,
            "error_type": "FI",
            "violated_rule": "format_outlier",
            "reason": (
                f"格式形态偏离该列主导形态（主导形态占比 {dom_ratio:.0%}，"
                f"本值形态 {value_shape(v)!r} 罕见）"
            ),
            "suggested_fix": "",
            "confidence": 0.8,
        })
    return errors
