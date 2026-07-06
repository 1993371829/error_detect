"""
检测器一：统计分布异常（文档 §8）。

- 数值列：Robust-Z |x-median|/(1.4826*MAD) 与 IQR 双判据（在近似干净子集上估参）。
- 日期列：解析为时间戳后用 Robust-Z 检出过早/未来等离群日期。
仅用 numpy/pandas，无第三方依赖。低频但合法的极值由后续融合/Stage3 兜底。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.coltypes import robust_stats
from stage_2.detectors.base import DetectorContext, clean_positions
from stage_2.encoding import _extract_number
from stage_2.schema import CandidateError


def _numeric_values(series: pd.Series) -> np.ndarray:
    """逐行单位感知数值（不可解析/空 -> nan），长度与 series 对齐。"""
    out = np.full(len(series), np.nan, dtype=np.float64)
    vals = series.astype(str).tolist()
    for i, v in enumerate(vals):
        if is_blank(v):
            continue
        out[i] = _extract_number(v)
    return out


# 常见日期格式：优先用显式 format 走 pandas 向量化 C 路径（快且不触发 dateutil 逐元素回退警告）
_COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y%m%d",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%Y-%m",
]


def _best_date_format(sample: pd.Series) -> Optional[str]:
    """在样本上挑选解析率最高的显式格式（>=0.8 才采用），否则 None（回退灵活解析）。"""
    best_fmt, best_rate = None, 0.0
    for fmt in _COMMON_DATE_FORMATS:
        rate = float(pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean())
        if rate > best_rate:
            best_fmt, best_rate = fmt, rate
        if best_rate >= 0.99:
            break
    return best_fmt if best_rate >= 0.8 else None


def _datetime_values(series: pd.Series) -> tuple[np.ndarray, float]:
    """逐行时间戳（秒，nan 表示不可解析）与解析率。

    先在样本上推断主导日期格式，命中则整列用该 format 向量化解析（快、无警告）；
    未命中再回退到灵活解析。仅解析整列一次。
    """
    non_blank_mask = ~series.map(is_blank)
    if not bool(non_blank_mask.any()):
        return np.full(len(series), np.nan), 0.0
    s_str = series.astype(str).where(non_blank_mask)
    non_blank = s_str[non_blank_mask]
    fmt = _best_date_format(non_blank.head(200))
    if fmt is not None:
        full = pd.to_datetime(s_str, format=fmt, errors="coerce")
    else:
        full = pd.to_datetime(s_str, errors="coerce")  # 回退灵活解析（较慢）
    ratio = float(full[non_blank_mask].notna().mean())
    ts = full.to_numpy().astype("datetime64[ns]").astype(np.int64).astype(np.float64)
    ts[full.isna().to_numpy()] = np.nan
    ts = ts / 1e9
    return ts, ratio


def detect_statistical(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    robust_z: float = 3.0,
    iqr_k: float = 1.5,
    min_clean: int = 20,
    detect_dates: bool = True,
) -> list[CandidateError]:
    cands: list[CandidateError] = []
    for col in df.columns:
        col = str(col)
        kind = ctx.kinds.get(col, "categorical")
        series = df[col]
        is_date = False
        if kind == "numeric":
            values = _numeric_values(series)
        elif detect_dates and kind in ("categorical", "highcard"):
            ts, ratio = _datetime_values(series)
            if ratio < 0.9:
                continue
            values = ts
            is_date = True
        else:
            continue

        clean_pos = clean_positions(clean_mask, col, series)
        clean_vals = values[clean_pos]
        clean_vals = clean_vals[~np.isnan(clean_vals)]
        if len(clean_vals) < min_clean:
            valid_all = values[~np.isnan(values)]
            if len(valid_all) < min_clean:
                continue
            clean_vals = valid_all

        med, mad, q1, q3 = robust_stats(clean_vals)
        iqr = max(q3 - q1, 1e-9)
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        fix_repr = _format_fix(med, is_date)

        for pos in range(len(values)):
            x = values[pos]
            if np.isnan(x):
                continue
            z = abs(x - med) / mad
            out_iqr = x < lo or x > hi
            if z <= robust_z and not out_iqr:
                continue
            evi = (
                f"{'日期' if is_date else '数值'}离群: robust_z={z:.2f}"
                f"(阈值{robust_z}), 中位数={fix_repr}"
            )
            cands.append(CandidateError(
                row_id=int(df.index[pos]),
                column=col,
                value=df.iloc[pos][col],
                detector="statistical",
                error_type="DIST",
                score=float(z),
                evidence=evi,
                suggested_fix=fix_repr,
                metadata={"robust_z": float(z), "is_date": is_date},
            ))
    return cands


def _format_fix(med: float, is_date: bool) -> str:
    if is_date:
        try:
            return pd.to_datetime(med, unit="s").strftime("%Y-%m-%d")
        except (ValueError, OverflowError, OSError):
            return ""
    return f"{med:.6g}"
