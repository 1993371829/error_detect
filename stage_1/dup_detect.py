"""
重复值检测：识别"整值由同一内容重复拼接而成"的单元格（如 'X,X' / 'X,X,X'）。

典型场景（movies Creator）：'Stephen King' 被写成 'Stephen King,Stephen King'。
这类错误格式合法、长度往往也在范围内，规则层（仅长度/正则）与分布层均难发现，
但结构上是确定的冗余复制。

判定（严格、低误报）：按分隔符切分后 >=2 个非空 token，且所有 token 归一化后完全相同。
该定义只命中"整值翻倍"，不会误伤"列表中个别项重复"（如地名列表里同一国家出现两次），
经 movies/rayyan/flights/beers/hospital/billionaire 验证：精度 0.99+，对无此类错误的数据集零标记。
"""

from __future__ import annotations

import pandas as pd

from stage_1.profiling import is_blank


def duplicate_fix(value: str, sep: str = ",") -> str | None:
    """若 value 为同一 token 重复拼接，返回去重后的单值；否则返回 None。"""
    toks = [t.strip() for t in str(value).split(sep)]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return None
    if len(set(t.lower() for t in toks)) == 1:
        return toks[0]
    return None


def detect_duplicates(series: pd.Series, column: str, *, sep: str = ",") -> list[dict]:
    """扫描单列，返回"整值重复拼接"的候选错误（交 Stage 3 复核）。"""
    errors: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        fix = duplicate_fix(str(value), sep)
        if fix is None:
            continue
        errors.append({
            "row_id": row_id,
            "column": column,
            "value": str(value),
            "error_type": "FI",
            "violated_rule": "duplicate_value",
            "reason": f"值由同一内容重复拼接而成，应为单值 {fix!r}",
            "suggested_fix": fix,
            "confidence": 0.9,
        })
    return errors
