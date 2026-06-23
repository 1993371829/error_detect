"""
检测器共享上下文与工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank


@dataclass
class DetectorContext:
    """各检测器共享的只读上下文（避免重复计算）。"""

    kinds: dict                          # 列 -> numeric/categorical/id/highcard
    semantic_types: dict = field(default_factory=dict)
    encoder: Optional[object] = None     # 已 fit 的 TabularEncoder（KNN/聚类用）
    x_all: Optional[np.ndarray] = None   # encoder.transform(df)
    row_clean: Optional[np.ndarray] = None  # 整行干净布尔（训练子集）


def column_clean_mask(clean_mask: Optional[pd.DataFrame], col: str, n: int) -> np.ndarray:
    """取某列干净布尔掩码（缺列/缺掩码则全 True）。"""
    if clean_mask is not None and col in clean_mask.columns:
        return clean_mask[col].astype(bool).to_numpy()
    return np.ones(n, dtype=bool)


def clean_positions(clean_mask: Optional[pd.DataFrame], col: str, series: pd.Series) -> np.ndarray:
    """该列“干净且非空”的行位置（用于在近似干净数据上估参）。"""
    n = len(series)
    col_clean = column_clean_mask(clean_mask, col, n)
    non_blank = ~series.map(is_blank).to_numpy()
    return np.where(col_clean & non_blank)[0]
