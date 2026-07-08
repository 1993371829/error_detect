"""
检测器：数值列类型/格式违规（数值元数一致性）。

动机（补统计/形态检测器的盲区）:
    数值列的一类高发注入错误是"把单个数值破坏成多值或带分隔符的串"，例如
    First Shown Year '2011' -> '2010 2011 2012'、RatingValue '6.1' -> '5/10,5/10'。
    - stage_2/detectors/statistical.py 的 _extract_number 要求整串是"单个数(+可选单位)"，
      这类串解析为 NaN 被直接丢弃、不报错（于是漏检，实测 First Shown Year 召回仅 0.02）。
    - stage_2/detectors/pattern_outlier.py 明确跳过数值列。
    两边都漏。

判据（无监督，在干净子集上估计"数值元数"）:
    1. 仅对"预判为数值"的列生效：干净非空格中可被 _extract_number 解析为单个数的比例 >= numeric_ratio_min。
    2. 统计干净格的"数值 token 个数"(用 _NUM_RE 计数)，取主导个数 dom_count 及其占比 dom_share。
    3. dom_share < min_dominant_share 则跳过（放过 ReviewCount/RatingCount 等天然多 token 列，控误报）。
    4. 非空格若 token 数 != dom_count（多值/缺值/掺入其它数），标为类型/格式违规候选(FI)。

高精度（数值列里出现异常数目的数值 token 几乎必为错误），误报交由融合与 Stage 3 兜底。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.detectors.base import DetectorContext, clean_positions
from stage_2.encoding import _NUM_RE, _extract_number
from stage_2.schema import CandidateError


def _num_tokens(val: str) -> int:
    """字符串中的数值 token 个数（如 '2010 2011 2012'->3, '6.1'->1, '90 min'->1）。"""
    return len(_NUM_RE.findall(val))


def detect_numeric_format(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    min_dominant_share: float = 0.9,
    numeric_ratio_min: float = 0.9,
    min_clean: int = 30,
) -> list[CandidateError]:
    """数值列类型/格式违规检测，返回候选错误列表（error_type=FI）。"""
    cands: list[CandidateError] = []
    for col in df.columns:
        col = str(col)
        kind = ctx.kinds.get(col)
        if kind == "empty":
            continue
        series = df[col]
        n = len(series)

        clean_pos = clean_positions(clean_mask, col, series)
        if len(clean_pos) < min_clean:
            clean_pos = np.where(~series.map(is_blank).to_numpy())[0]
        if len(clean_pos) < min_clean:
            continue
        clean_vals = series.iloc[clean_pos].astype(str)

        # 数值列判定：干净非空格中可解析为"单个数(+可选单位)"的比例足够高。
        numeric_ratio = float(np.mean([
            not np.isnan(_extract_number(v)) for v in clean_vals
        ]))
        if numeric_ratio < numeric_ratio_min:
            continue

        tok_counts = Counter(_num_tokens(v) for v in clean_vals)
        total = sum(tok_counts.values())
        if total == 0:
            continue
        dom_count, dom_c = tok_counts.most_common(1)[0]
        dom_share = dom_c / total
        if dom_share < min_dominant_share:
            continue

        for pos in range(n):
            val = series.iloc[pos]
            if is_blank(val):
                continue
            tc = _num_tokens(str(val))
            if tc == dom_count:
                continue
            cands.append(CandidateError(
                row_id=int(df.index[pos]), column=col, value=val,
                detector="numeric_format", error_type="FI",
                score=0.9,
                evidence=f"数值列元数违规：期望 {dom_count} 个数值(占比 {dom_share:.0%})，"
                         f"实际 {tc} 个",
                suggested_fix=None,
                metadata={"expected_tokens": dom_count, "actual_tokens": tc,
                          "subtype": "numeric_arity"},
            ))
    return cands
