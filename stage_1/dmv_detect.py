"""
伪缺失值检测（DMV, Disguised Missing Value）。

借鉴 Cocoon：有些单元格并非真正为空，但其取值在语义上表示"缺失/未知/不适用"
（如 "?"、"unknown"、"missing"、"n.a."、"tbd"、占位数字 "9999" 等）。这类值不会
被独立的 MV 扫描（空串/NaN/标准缺失哨兵）捕获，需要基于词表 + 画像额外识别。

纯词表/统计匹配，不调用 LLM。真正的空值（is_blank）由 MV 扫描负责，这里只看非空值，
因此与 MV 不会重复标记（profiling.MISSING_SENTINELS 中的哨兵已被算作 MV）。
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from stage_1.profiling import is_blank

# 默认伪缺失词表（匹配前会做 strip + lower）。
# 注意：不包含已被 MV 扫描覆盖的标准缺失哨兵
# （""/empty/null/n/a/na/none/-，见 profiling.MISSING_SENTINELS）。
DEFAULT_DMV_TOKENS = frozenset({
    "?", "??", "???",
    "unknown", "unk", "n.a.", "n.a", "nil", "nan",
    "missing", "tbd", "tba", "undefined", "unspecified",
    "not available", "not applicable", "no value", "no data",
    "xx", "xxx", "xxxx", "--", "---", "...",
})

# 常见占位数字（仅在 detect_numeric_placeholder=True 时生效）。
# 数值占位易误伤合法值（如 0），默认关闭，需用户显式开启。
DEFAULT_NUMERIC_PLACEHOLDERS = frozenset({"9999", "99999", "999999", "-1", "-999"})


def detect_dmv(
    series: pd.Series,
    column: str,
    *,
    extra_tokens: Optional[Iterable[str]] = None,
    detect_numeric_placeholder: bool = False,
    numeric_placeholders: Optional[Iterable[str]] = None,
) -> list[dict]:
    """
    在单列内检测伪缺失值（DMV）。

    判定逻辑（精确匹配，非子串）:
        - 单元格非空（is_blank=False）；
        - 其 strip+lower 后的整值精确等于某个伪缺失 token；
        - 或（可选）其 strip 后的整值精确等于某个占位数字。

    采用"整值精确匹配"而非子串匹配，避免误伤把 token 作为正文一部分的自由文本
    （如地址里出现 "missing st." 不会被命中）。

    Args:
        series: 待检测列
        column: 列名（用于输出）
        extra_tokens: 在默认词表外追加的伪缺失 token
        detect_numeric_placeholder: 是否检测占位数字（默认关闭，控误报）
        numeric_placeholders: 自定义占位数字集合（默认 DEFAULT_NUMERIC_PLACEHOLDERS）

    Returns:
        错误记录 dict 列表，每条 error_type=DMV，含 suggested_fix(空串，交由后续标准化)
        与 confidence。
    """
    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return []

    tokens = {str(t).strip().lower() for t in DEFAULT_DMV_TOKENS}
    if extra_tokens:
        tokens |= {str(t).strip().lower() for t in extra_tokens}

    num_ph: set[str] = set()
    if detect_numeric_placeholder:
        num_ph = {str(x).strip() for x in (numeric_placeholders or DEFAULT_NUMERIC_PLACEHOLDERS)}

    errors: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        sval = str(value)
        norm = sval.strip().lower()
        hit = None
        if norm in tokens:
            hit = "token"
        elif num_ph and sval.strip() in num_ph:
            hit = "numeric_placeholder"
        if hit is None:
            continue
        errors.append({
            "row_id": row_id,
            "column": column,
            "value": sval,
            "error_type": "DMV",
            "violated_rule": "disguised_missing_value",
            "reason": (
                f"疑似伪缺失值: '{sval}' 语义上表示缺失/未知"
                + ("（占位数字）" if hit == "numeric_placeholder" else "")
            ),
            "suggested_fix": "",
            "confidence": 0.8,
        })
    return errors


def detect_dmv_dataframe(df: pd.DataFrame, **kwargs) -> list[dict]:
    """对 DataFrame 每列执行 DMV 检测并汇总结果。"""
    errors: list[dict] = []
    for col in df.columns:
        errors.extend(detect_dmv(df[col], str(col), **kwargs))
    return errors
