"""
基于语义类型的硬范围规则（文档 §5.3）。

依据列 semantic_type（来自 LLM 归纳）施加高精度硬约束，违反即记为 FI/range_error。
关键词匹配 semantic_type / 列名，仅对可解析为数值的值施加，规则保守、误报率低。
"""

from __future__ import annotations

import math

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.encoding import _extract_number

# (关键词集合, (下界 or None, 上界 or None, 是否要求整数), 说明)
_RANGES = [
    ({"age"}, (0.0, 120.0, False), "年龄应在 0~120"),
    ({"percent", "percentage", "rate", "ratio"}, (0.0, 100.0, False), "百分比应在 0~100"),
    ({"latitude"}, (-90.0, 90.0, False), "纬度应在 -90~90"),
    ({"longitude"}, (-180.0, 180.0, False), "经度应在 -180~180"),
    ({"price", "amount", "cost", "salary", "income", "currency", "fee"},
     (0.0, None, False), "金额不应为负"),
    ({"quantity", "count", "number", "qty"}, (0.0, None, True), "数量应为非负整数"),
    ({"year"}, (1000.0, 2100.0, True), "年份应在合理区间"),
]


def _match_range(semantic_type: str, column: str):
    text = f"{semantic_type} {column}".lower()
    for keywords, bounds, reason in _RANGES:
        if any(k in text for k in keywords):
            return bounds, reason
    return None


def detect_range_errors(
    series: pd.Series,
    column: str,
    semantic_type: str,
    *,
    numeric_min_ratio: float = 0.8,
) -> list[dict]:
    matched = _match_range(semantic_type or "", column)
    if matched is None:
        return []
    (lo, hi, need_int), reason = matched

    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return []
    parsed = non_blank.map(_extract_number)
    if float(parsed.notna().mean()) < numeric_min_ratio:
        return []  # 该列实际不是数值，放弃（避免误判）

    errors: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        x = _extract_number(str(value))
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        bad = (lo is not None and x < lo) or (hi is not None and x > hi)
        if not bad and need_int and abs(x - round(x)) > 1e-9:
            bad = True
        if not bad:
            continue
        errors.append({
            "row_id": row_id, "column": column, "value": value,
            "error_type": "FI", "violated_rule": "range_error",
            "reason": f"{reason}（语义类型 {semantic_type}），当前值 {value}",
            "confidence": 0.95,
        })
    return errors
