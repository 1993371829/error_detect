"""
检测器七：聚类 / 局部密度异常（文档 §14）。

行级异常（LOF + KMeans 距离），再用逐列 robust-z（数值）/ 低频（类别）定位异常列。
权重低（行级方法定位列不可靠）。默认关闭。
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


def detect_clustering(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    lof_neighbors: int = 20,
    contamination: float = 0.05,
    max_cols_per_row: int = 1,
    min_rows: int = 50,
) -> list[CandidateError]:
    if ctx.encoder is None or ctx.x_all is None:
        return []
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        return []
    x_all = ctx.x_all
    n = len(df)
    if n < min_rows:
        return []

    lof = LocalOutlierFactor(
        n_neighbors=min(lof_neighbors, n - 1), contamination=contamination,
    )
    labels = lof.fit_predict(x_all)              # -1 = 离群
    outlier_rows = np.where(labels == -1)[0]
    if len(outlier_rows) == 0:
        return []

    # 预备逐列定位所需统计
    num_vals = {c: _numeric_values(df[c]) for c in df.columns
                if ctx.kinds.get(str(c)) == "numeric"}
    num_stats = {}
    for c, v in num_vals.items():
        clean = v[~np.isnan(v)]
        if len(clean):
            num_stats[c] = robust_stats(clean)
    cat_freq = {}
    for c in df.columns:
        if ctx.kinds.get(str(c)) == "categorical":
            nb = df[c][~df[c].map(is_blank)].astype(str)
            cat_freq[str(c)] = Counter(nb)

    cands: list[CandidateError] = []
    for pos in outlier_rows:
        scored: list[tuple[float, str, object, str]] = []
        for col in df.columns:
            col = str(col)
            val = df.iloc[pos][col]
            if is_blank(val):
                continue
            kind = ctx.kinds.get(col)
            if kind == "numeric" and col in num_stats:
                x = num_vals[col][pos]
                if np.isnan(x):
                    continue
                med, mad, _, _ = robust_stats(num_vals[col][~np.isnan(num_vals[col])])
                z = abs(x - med) / mad
                scored.append((z, col, val, f"行级离群且该列 robust_z={z:.2f}"))
            elif kind == "categorical" and col in cat_freq:
                cnt = cat_freq[col]
                total = sum(cnt.values()) or 1
                freq = cnt.get(str(val), 0) / total
                scored.append((1.0 - freq, col, val, f"行级离群且该列取值低频({freq:.3f})"))
        scored.sort(key=lambda t: t[0], reverse=True)
        for z, col, val, evi in scored[:max_cols_per_row]:
            if z <= 0:
                continue
            cands.append(CandidateError(
                row_id=int(df.index[pos]), column=col, value=val,
                detector="clustering", error_type="DIST",
                score=float(z), evidence=evi, suggested_fix=None,
                metadata={"row_outlier": True},
            ))
    return cands
