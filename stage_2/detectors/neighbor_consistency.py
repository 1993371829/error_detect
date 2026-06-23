"""
检测器五：相似记录一致性异常（文档 §12）。

用已 fit 的 TabularEncoder 把行向量化，在近似干净行上建 KNN；对每行的近邻：
    - 数值列：本值偏离近邻中位数的标准化程度 > 阈值 -> 候选（fix=近邻中位数）。
    - 类别列：近邻多数值占比 > 阈值且本值 != 多数值 -> 候选（fix=多数值）。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.coltypes import robust_stats
from stage_2.detectors.base import DetectorContext
from stage_2.detectors.statistical import _numeric_values
from stage_2.schema import CandidateError


def detect_neighbor_consistency(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    k: int = 10,
    num_deviation: float = 3.0,
    cat_majority: float = 0.8,
    min_clean_rows: int = 50,
) -> list[CandidateError]:
    if ctx.encoder is None or ctx.x_all is None or ctx.row_clean is None:
        return []
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return []

    x_all = ctx.x_all
    clean_idx = np.where(ctx.row_clean)[0]
    if len(clean_idx) < min_clean_rows:
        return []
    x_clean = x_all[clean_idx]
    k_eff = min(k + 1, len(clean_idx))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(x_clean)
    _, nbr = nn.kneighbors(x_all)            # (n, k_eff) -> 索引指向 clean_idx
    nbr_rows = clean_idx[nbr]                # 映射回原始行位置

    num_cache = {c: _numeric_values(df[c]) for c in df.columns
                 if ctx.kinds.get(str(c)) == "numeric"}

    cands: list[CandidateError] = []
    n = len(df)
    for pos in range(n):
        neigh = [r for r in nbr_rows[pos] if r != pos][:k]
        if len(neigh) < 3:
            continue
        for col in df.columns:
            col = str(col)
            kind = ctx.kinds.get(col)
            val = df.iloc[pos][col]
            if is_blank(val):
                continue
            if kind == "numeric":
                vals = num_cache[col][neigh]
                vals = vals[~np.isnan(vals)]
                if len(vals) < 3:
                    continue
                x = num_cache[col][pos]
                if np.isnan(x):
                    continue
                med, mad, _, _ = robust_stats(vals)
                dev = abs(x - med) / mad
                if dev > num_deviation:
                    cands.append(CandidateError(
                        row_id=int(df.index[pos]), column=col, value=val,
                        detector="neighbor_consistency", error_type="DIST",
                        score=float(dev),
                        evidence=f"{len(vals)} 个近邻该列中位数≈{med:.6g}，本值偏离 {dev:.2f}",
                        suggested_fix=f"{med:.6g}",
                        metadata={"deviation": float(dev)},
                    ))
            elif kind in ("categorical", "id"):
                vals = [str(df.iloc[r][col]) for r in neigh if not is_blank(df.iloc[r][col])]
                if len(vals) < 3:
                    continue
                cnt = Counter(vals)
                maj, maj_n = cnt.most_common(1)[0]
                share = maj_n / len(vals)
                if share >= cat_majority and str(val) != maj:
                    cands.append(CandidateError(
                        row_id=int(df.index[pos]), column=col, value=val,
                        detector="neighbor_consistency", error_type="VAD",
                        score=float(share),
                        evidence=f"{len(vals)} 个近邻该列 {share:.0%} 为 {maj!r}，本值为 {val!r}",
                        suggested_fix=maj,
                        metadata={"majority_share": float(share)},
                    ))
    return cands
