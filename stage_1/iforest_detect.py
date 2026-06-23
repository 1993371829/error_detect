"""
数值列联合孤立森林极端值检测（文档 §5.4）。

仅用于多数值列联合异常，取极端 top 分位（高精度）。孤立森林输出行异常，
再用逐列 robust-z 定位最异常的数值列，输出单元格级候选（FI/extreme_outlier）。
默认关闭：基准数据多为类别/文本列，按需开启。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.coltypes import infer_column_kinds, robust_stats
from stage_2.encoding import _extract_number


def _numeric_matrix(df: pd.DataFrame, num_cols: list[str]) -> np.ndarray:
    cols = []
    for c in num_cols:
        vals = df[c].astype(str).map(lambda v: np.nan if is_blank(v) else _extract_number(v))
        arr = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=np.float64)
        med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
        arr = np.where(np.isnan(arr), med, arr)
        cols.append(arr)
    return np.vstack(cols).T if cols else np.empty((len(df), 0))


def detect_iforest_outliers(
    df: pd.DataFrame,
    *,
    top_quantile: float = 0.995,
    min_numeric_cols: int = 2,
    min_rows: int = 50,
    random_state: int = 42,
) -> list[dict]:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
    except ImportError:
        return []
    if len(df) < min_rows:
        return []

    kinds = infer_column_kinds(df)
    num_cols = [str(c) for c in df.columns if kinds.get(str(c)) == "numeric"]
    if len(num_cols) < min_numeric_cols:
        return []

    X = _numeric_matrix(df, num_cols)
    Xs = RobustScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    model.fit(Xs)
    scores = -model.score_samples(Xs)               # 越大越异常
    thr = float(np.quantile(scores, top_quantile))
    outlier_rows = np.where(scores >= thr)[0]
    if len(outlier_rows) == 0:
        return []

    # 逐列 robust-z（在全列上估参）用于定位异常列
    col_stats = {}
    for j, c in enumerate(num_cols):
        col = X[:, j]
        med, mad, _, _ = robust_stats(col)
        col_stats[c] = (med, mad)

    errors: list[dict] = []
    for pos in outlier_rows:
        best_col, best_z = None, 0.0
        for j, c in enumerate(num_cols):
            med, mad = col_stats[c]
            z = abs(X[pos, j] - med) / mad
            if z > best_z:
                best_z, best_col = z, c
        if best_col is None or best_z <= 0:
            continue
        errors.append({
            "row_id": df.index[pos], "column": best_col,
            "value": df.iloc[pos][best_col],
            "error_type": "FI", "violated_rule": "extreme_outlier",
            "reason": f"孤立森林行级极端异常(score={scores[pos]:.3f})，"
                      f"该行 {best_col} robust_z={best_z:.2f} 最大",
            "confidence": 0.9,
        })
    return errors
