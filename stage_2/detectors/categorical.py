"""
检测器二：低频类别 / 拼写相似 / 格式簇异常（文档 §9）。

- 低频类别：count<=rare_max 或 freq<freq_floor 的取值作为候选（仅候选，非定论）。
- 拼写相似：低频值与某高频锚点 rapidfuzz 相似度 > sim_threshold -> typo，给 suggested_fix。
- 格式簇：值的字符模式在列内极罕见 -> 格式异常（低分，无修复）。
高召回，误报交融合与 Stage 3 兜底。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

from stage_1.profiling import is_blank
from stage_2.detectors.base import DetectorContext, column_clean_mask
from stage_2.schema import CandidateError


def _to_pattern(s: str) -> str:
    out = []
    for c in s:
        if c.isdigit():
            out.append("d")
        elif c.isupper():
            out.append("L")
        elif c.islower():
            out.append("l")
        else:
            out.append(c)
    return "".join(out)


def detect_categorical(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    max_unique: int = 1000,
    rare_max_count: int = 2,
    freq_floor: float = 0.001,
    min_anchor_count: int = 3,
    sim_threshold: float = 0.85,
    pattern_floor: float = 0.02,
    min_rows: int = 30,
) -> list[CandidateError]:
    cands: list[CandidateError] = []
    for col in df.columns:
        col = str(col)
        if ctx.kinds.get(col) != "categorical":
            continue
        series = df[col]
        n = len(series)
        col_clean = column_clean_mask(clean_mask, col, n)
        clean_vals = series[col_clean]
        clean_nb = clean_vals[~clean_vals.map(is_blank)].astype(str)
        if len(clean_nb) < min_rows:
            clean_nb = series[~series.map(is_blank)].astype(str)
        if len(clean_nb) < min_rows:
            continue
        counts = Counter(clean_nb)
        if len(counts) > max_unique:
            continue
        total = sum(counts.values())

        anchors = [v for v, c in counts.items() if c >= min_anchor_count]
        pattern_counts = Counter(_to_pattern(v) for v in clean_nb)
        pat_total = sum(pattern_counts.values())

        # 每个 distinct 值的判定缓存
        typo_fix: dict[str, Optional[str]] = {}
        for value in set(series[~series.map(is_blank)].astype(str)):
            c = counts.get(value, 0)
            freq = c / total if total else 0.0
            is_rare = c <= rare_max_count or freq < freq_floor
            if not is_rare or not anchors:
                typo_fix[value] = None
                continue
            match = process.extractOne(value, anchors, scorer=fuzz.ratio)
            if match and match[0] != value and (match[1] / 100.0) >= sim_threshold:
                typo_fix[value] = match[0]
            else:
                typo_fix[value] = None

        for pos in range(n):
            val = series.iloc[pos]
            if is_blank(val):
                continue
            val = str(val)
            fix = typo_fix.get(val)
            if fix is not None:
                sim = fuzz.ratio(val, fix) / 100.0
                cands.append(CandidateError(
                    row_id=int(df.index[pos]), column=col, value=val,
                    detector="categorical_typo", error_type="T",
                    score=float(sim),
                    evidence=f"低频值 '{val}' 与高频值 '{fix}' 相似度 {sim:.2f}",
                    suggested_fix=fix,
                    metadata={"similarity": float(sim)},
                ))
                continue
            # 格式簇异常（仅当无拼写匹配时）
            pat = _to_pattern(val)
            pat_freq = pattern_counts.get(pat, 0) / pat_total if pat_total else 0.0
            if pat_freq < pattern_floor:
                cands.append(CandidateError(
                    row_id=int(df.index[pos]), column=col, value=val,
                    detector="format_cluster", error_type="FI",
                    score=float(1.0 - pat_freq),
                    evidence=f"罕见字符模式 '{pat}'(列内占比 {pat_freq:.3f})",
                    suggested_fix=None,
                    metadata={"pattern": pat, "pattern_freq": float(pat_freq)},
                ))
    return cands
