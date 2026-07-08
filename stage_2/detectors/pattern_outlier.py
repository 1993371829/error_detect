"""
检测器：形态串 / 日期格式离群（补 rayyan/movies 等日期与格式化字段的召回缺口）。

动机:
    重构/统计/类别检测器对"高基数格式化字段"（如时间戳、日期、编号）覆盖不足：
    - 高基数列被 categorical 检测器跳过（唯一值过多）。
    - 数值统计对 'YYYY-MM-DD' 这类字符串无从下手。
    这类列往往有一个高度主导的字符形态（shape），少数不符合主流形态或不可解析
    的值即为格式错误（FI）。

判定（无监督，在干净子集上估计主流形态）:
    1. 把每个值抽象成形态串：数字->d、大写->L、小写->l，其余字符原样保留。
    2. 在干净单元格上统计形态串分布，取主流形态及其占比 dominant_share。
    3. 非日期列：仅当存在强主流形态（占比 >= dominant_share，排除自由文本）时，
       把"形态罕见（计数 <= rare_max 且不等于主流）"的值标为候选。
    4. 日期列（列名/semantic_type 提示，或高解析率）：主流形态即期望日期格式，
       形态不符主流 或 不可被解析为日期 的值标为候选。

高召回，误报交由融合与 Stage 3 兜底。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.detectors.base import DetectorContext, clean_positions
from stage_2.schema import CandidateError

# 列名中出现这些关键词则视为日期/时间字段（额外做可解析性校验）
_DATE_NAME_HINTS = (
    "date", "time", "year", "created", "updated", "published",
    "birth", "day", "month", "timestamp", "_at", "dob",
)


def _shape(s: str) -> str:
    """把字符串抽象为形态串：数字->d、大写->L、小写->l，其余原样保留。"""
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


def _looks_like_date(col: str, ctx: DetectorContext, clean_vals: pd.Series) -> bool:
    """判断列是否为日期/时间字段：列名关键词 或 semantic_type 提示 或 高解析率。"""
    name = col.lower()
    if any(h in name for h in _DATE_NAME_HINTS):
        return True
    sem = str(ctx.semantic_types.get(col, "") or "").lower()
    if "date" in sem or "time" in sem or "year" in sem:
        return True
    # 采样解析率：>= 0.8 可解析为日期则视为日期列（限制样本量控成本）
    sample = clean_vals.head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.8


def _parseable_date(val: str) -> bool:
    try:
        return not pd.isna(pd.to_datetime(val, errors="coerce", format="mixed"))
    except (ValueError, TypeError):
        return False


def detect_pattern_outlier(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    dominant_share: float = 0.8,
    rare_max: int = 2,
    min_rows: int = 30,
) -> list[CandidateError]:
    """形态串 / 日期格式离群检测，返回候选错误列表（error_type=FI）。"""
    cands: list[CandidateError] = []
    for col in df.columns:
        col = str(col)
        # 纯数值列的离群交给 statistical 检测器；空列跳过。
        kind = ctx.kinds.get(col)
        if kind in ("numeric", "empty"):
            continue
        series = df[col]
        n = len(series)

        clean_pos = clean_positions(clean_mask, col, series)
        if len(clean_pos) < min_rows:
            clean_pos = np.where(~series.map(is_blank).to_numpy())[0]
        if len(clean_pos) < min_rows:
            continue
        clean_vals = series.iloc[clean_pos].astype(str)

        pat_counts = Counter(_shape(v) for v in clean_vals)
        pat_total = sum(pat_counts.values())
        if pat_total == 0:
            continue
        dom_pat, dom_c = pat_counts.most_common(1)[0]
        dom_share = dom_c / pat_total

        is_date = _looks_like_date(col, ctx, clean_vals)
        # 关键修复：无强主流形态（自由文本 / 多合法格式并存，如 '8 September 1960 (USA)'）一律跳过。
        # 日期列同样受此闸门约束——此前日期列绕过该闸门，导致主流占比极低(如 14%)的多格式日期列
        # 整列被判"格式不符"，制造海量误报。只有形态确实统一(dom_share 达标)的列才做离群判定。
        if dom_share < dominant_share:
            continue

        # 日期列缓存逐值解析结果，避免重复解析同一取值
        parse_cache: dict[str, bool] = {}
        subtype = "date" if is_date else "shape"
        for pos in range(n):
            val = series.iloc[pos]
            if is_blank(val):
                continue
            val = str(val)
            pat = _shape(val)

            if pat == dom_pat:
                # 形态与主流一致：日期列再校验可解析性（形态对但内容非法日期）。
                if is_date:
                    if val not in parse_cache:
                        parse_cache[val] = _parseable_date(val)
                    if not parse_cache[val]:
                        cands.append(CandidateError(
                            row_id=int(df.index[pos]), column=col, value=val,
                            detector="pattern_outlier", error_type="FI", score=0.85,
                            evidence=f"不可解析为有效日期：'{val}' 形态 '{pat}'，主流 "
                                     f"'{dom_pat}'({dom_share:.0%})",
                            suggested_fix=None,
                            metadata={"pattern": pat, "dominant": dom_pat, "subtype": subtype},
                        ))
                continue

            # 形态与主流不同：仅当该形态罕见（cnt <= rare_max）才作候选，控误报。
            cnt = pat_counts.get(pat, 0)
            if cnt > rare_max:
                continue
            score = 1.0 - cnt / pat_total
            reason = "日期格式不符主流" if is_date else "罕见形态"
            cands.append(CandidateError(
                row_id=int(df.index[pos]), column=col, value=val,
                detector="pattern_outlier", error_type="FI",
                score=float(max(score, 0.6)),
                evidence=f"{reason} '{pat}'(计数 {cnt})，主流 '{dom_pat}'({dom_share:.0%})",
                suggested_fix=None,
                metadata={"pattern": pat, "dominant": dom_pat, "subtype": subtype},
            ))
    return cands
