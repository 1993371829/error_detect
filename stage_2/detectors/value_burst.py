"""
检测器：异常高频重复值（默认值/占位值注入嫌疑）。

针对"值本身格式合法且是高频多数派"因而被其余检测器全部放过的系统性错误
（如 rayyan article_jcreated_at 中重复数百次的 `1/1/14` 默认日期）。

两条互补路径（都要求列为高多样性 categorical/highcard，distinct_ratio 闸门
排除类别列的天然高频；id 类列另有 predictability 闸门兜底）:
    1. 日期样式列：正常时间戳/日期近乎唯一，任何 count>=burst_min_count 且
       share>=burst_min_share 的重复值都是"默认日期"嫌疑（6 数据集模拟：仅
       rayyan created_at 命中，337 格全为 GT 真错）。
    2. 孤立尖峰：top1 频次断崖式高于 top2（>= spike_top2_ratio 倍），典型如
       author_list 的 '{NULL}' (63 次 vs 次高 2 次)。真实热门值列（movies 的
       'Los Angeles...' 350 vs 135）头部平滑衰减，不会命中。

低权重进融合层，最终由 Stage 3 结合证据裁决。
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.detectors.base import DetectorContext
from stage_2.schema import CandidateError

_DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def detect_value_burst(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    min_rows: int = 100,
    min_distinct_ratio: float = 0.15,
    date_like_ratio: float = 0.6,
    burst_min_count: int = 15,
    burst_min_share: float = 0.03,
    spike_min_count: int = 20,
    spike_min_share: float = 0.05,
    spike_top2_ratio: float = 10.0,
) -> list[CandidateError]:
    cands: list[CandidateError] = []
    for col in df.columns:
        col = str(col)
        if ctx.kinds.get(col) not in ("categorical", "highcard"):
            continue
        series = df[col]
        non_blank = series[~series.map(is_blank)].astype(str)
        n = len(non_blank)
        if n < min_rows:
            continue
        vc = non_blank.value_counts()
        distinct_ratio = vc.shape[0] / n
        if distinct_ratio < min_distinct_ratio:
            continue

        burst_values: dict[str, str] = {}  # 值 -> 证据说明

        # 路径 1：日期样式列的任何高频重复值
        if float(non_blank.map(lambda v: bool(_DATE_RE.match(v))).mean()) >= date_like_ratio:
            for val, cnt in vc.items():
                share = cnt / n
                if cnt < burst_min_count or share < burst_min_share:
                    break  # vc 降序，后续更小
                burst_values[str(val)] = (
                    f"日期列多样性高(distinct_ratio={distinct_ratio:.2f})，正常取值应近乎唯一，"
                    f"但该值重复 {cnt} 次(占比 {share:.1%})，疑为默认日期注入"
                )

        # 路径 2：孤立尖峰（top1 断崖式高于 top2）
        if not burst_values and vc.shape[0] >= 2:
            top1, top2 = int(vc.iloc[0]), int(vc.iloc[1])
            share1 = top1 / n
            if (
                top1 >= spike_min_count
                and share1 >= spike_min_share
                and top1 >= spike_top2_ratio * max(top2, 1)
            ):
                val = str(vc.index[0])
                burst_values[val] = (
                    f"高多样性列(distinct_ratio={distinct_ratio:.2f})中孤立尖峰："
                    f"该值出现 {top1} 次(占比 {share1:.1%})，是次高频值({top2} 次)的 "
                    f"{top1 / max(top2, 1):.0f} 倍，疑为占位/默认值注入"
                )

        if not burst_values:
            continue
        for pos in range(len(series)):
            val = series.iloc[pos]
            if is_blank(val):
                continue
            val = str(val)
            evidence = burst_values.get(val)
            if evidence is None:
                continue
            cnt = int(vc.get(val, 0))
            cands.append(CandidateError(
                row_id=int(df.index[pos]), column=col, value=val,
                detector="value_burst", error_type="VAD",
                score=float(min(cnt / n / burst_min_share, 1.0)),
                evidence=evidence,
                suggested_fix=None,
                metadata={"count": cnt, "share": float(cnt / n)},
            ))
    return cands
