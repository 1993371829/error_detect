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

import re
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


def _coarse_shape(s: str) -> str:
    """粗化形态：连续字母块 -> A、连续数字块 -> N，其余原样。

    月份名长短、单双位日期等细节被归并，同一日期格式收敛为一个簇
    （'19 December 1969 (USA)' 与 '8 May 2001 (USA)' 同为 'N A N (A)'）。
    """
    s = re.sub(r"[A-Za-z]+", "A", s)
    s = re.sub(r"\d+", "N", s)
    return s


def _is_subsequence(sub: str, full: str) -> bool:
    """sub 是否为 full 的字符子序列（保序、可不连续）。

    合法的日期粒度变体通常是主导格式的"截断"（'A N (A)' ⊂ 'N A N (A)'，即缺日的
    'May 1985 (USA)'）；而结构重排（'A N, N A'）或加长（'N:NA N'）不是子序列，
    属于另一套写法，是格式不一致的强信号。
    """
    it = iter(full)
    return all(c in it for c in sub)


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
    secondary_dominant_share: float = 0.6,
    secondary_min_share: float = 0.02,
    secondary_max_share: float = 0.40,
    secondary_min_count: int = 20,
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

        # 次级格式簇路径（仅日期/时间列）：细形态因月份名长短等被打散（Release Date 细形态
        # 主导仅 14%），改用粗形态（字母块->A/数字块->N）聚簇。当存在明显主导格式时，占比
        # 5%~40% 级别的"次级格式簇"（如 'Apr 17, 1981 Wide' vs 主导 '17 April 1981 (USA)'）
        # 高于 rare 阈值而被细形态路径放过——此处以低分交融合层与 Stage 3 裁决
        # （prompt 已有"多种写法应判 FI"原则）。仅限日期列：非日期列（人名/多值列表等）
        # 的次级粗形态多为天然合法变体，6 数据集模拟显示误报过高。
        if is_date:
            coarse_counts = Counter(_coarse_shape(v) for v in clean_vals)
            cdom_pat, cdom_c = coarse_counts.most_common(1)[0]
            if cdom_c / pat_total >= secondary_dominant_share:
                # 子序列过滤：截断粒度变体（'A N (A)' ⊂ 'N A N (A)'，如缺日的
                # 'May 1985 (USA)'）多为合法写法（movies 实测 344/427 合法），跳过；
                # 结构重排（'Apr 17, 1981 Wide'）或加长（'7:10aDec 1'）才是格式不一致。
                secondary = {
                    p for p, c in coarse_counts.items()
                    if p != cdom_pat
                    and c >= secondary_min_count
                    and secondary_min_share <= c / pat_total <= secondary_max_share
                    and not _is_subsequence(p, cdom_pat)
                }
                if secondary:
                    for pos in range(n):
                        val = series.iloc[pos]
                        if is_blank(val):
                            continue
                        val = str(val)
                        cpat = _coarse_shape(val)
                        if cpat not in secondary:
                            continue
                        c = coarse_counts.get(cpat, 0)
                        cands.append(CandidateError(
                            row_id=int(df.index[pos]), column=col, value=val,
                            detector="secondary_format", error_type="FI",
                            score=0.6,
                            evidence=f"次级日期格式簇 '{cpat}'(占比 {c / pat_total:.1%})，"
                                     f"主导格式 '{cdom_pat}'({cdom_c / pat_total:.0%})——"
                                     f"同列存在两种系统性写法，疑为格式不一致",
                            suggested_fix=None,
                            metadata={"pattern": cpat, "dominant": cdom_pat,
                                      "subtype": "secondary_format"},
                        ))

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
