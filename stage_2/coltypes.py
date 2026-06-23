"""
列类型/语义推断的统一访问层（供 Stage 2 各检测器与 Stage 1 硬范围规则共享）。

复用 stage_1 画像口径与 stage_2/encoding 的单位感知数值解析，避免各检测器
各自造画像。kind 取值：numeric / categorical / id / highcard / datetime / empty。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.encoding import _extract_number

_MAD_SCALE = 1.4826


def parse_numeric(series: pd.Series) -> pd.Series:
    """单位感知数值解析（与 encoding 一致），非空且不可解析为 NaN。"""
    non_blank = series[~series.map(is_blank)].astype(str)
    return non_blank.map(_extract_number)


def numeric_ratio(series: pd.Series) -> float:
    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return 0.0
    parsed = non_blank.map(_extract_number)
    return float(parsed.notna().mean())


def robust_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """返回 (median, mad_scaled, q1, q3)；mad_scaled 已乘 1.4826，且有下限。"""
    if len(values) == 0:
        return 0.0, 1.0, 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * _MAD_SCALE
    q1, q3 = float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))
    if mad <= 1e-9:
        # MAD 退化（多数值相同）时退回标准差，再退回 IQR
        std = float(np.std(values))
        mad = std if std > 1e-9 else max((q3 - q1), 1.0)
    return med, mad, q1, q3


def infer_column_kinds(
    df: pd.DataFrame,
    *,
    numeric_min_ratio: float = 0.8,
    max_cardinality: int = 500,
    id_min_len: int = 4,
) -> dict[str, str]:
    """
    粗分每列类型，口径与 stage_2/encoding 对齐：
        - id：纯整数 + 串长稳定且较长（ProviderNumber/ZipCode）
        - numeric：单位感知可解析比例 >= numeric_min_ratio（且非 id）
        - categorical：唯一值 <= max_cardinality
        - highcard：超高基数自由文本
        - empty：全空
    """
    kinds: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        non_blank = series[~series.map(is_blank)].astype(str)
        if non_blank.empty:
            kinds[str(col)] = "empty"
            continue
        lengths = non_blank.str.len()
        nunique = int(non_blank.nunique())
        plain = pd.to_numeric(non_blank, errors="coerce")
        is_id = bool(
            plain.notna().mean() >= numeric_min_ratio
            and (plain.dropna() % 1 == 0).all()
            and float(lengths.median()) >= id_min_len
            and float(lengths.std() or 0.0) <= 1.0
        )
        if is_id:
            kinds[str(col)] = "id"
            continue
        if float(non_blank.map(_extract_number).notna().mean()) >= numeric_min_ratio:
            kinds[str(col)] = "numeric"
            continue
        kinds[str(col)] = "categorical" if nunique <= max_cardinality else "highcard"
    return kinds


def load_semantic_types(rules_path: str | Path) -> dict[str, str]:
    """从 Stage1 rules.json 读取 列 -> semantic_type（缺失返回空 dict）。"""
    path = Path(rules_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, str] = {}
    for entry in rules:
        if isinstance(entry, dict) and "column" in entry:
            out[str(entry["column"])] = str(entry.get("semantic_type", "") or "")
    return out
