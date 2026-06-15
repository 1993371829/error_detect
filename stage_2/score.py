"""
Stage 2 打分与输出：把模型的逐特征误差还原成"逐列贡献"，按无监督阈值
筛出可疑单元格，产出可并入 Stage 1 / 送往 Stage 3 的候选错误。

角色分工:
    - DAE 走"单元格级"通道：逐列误差经按列鲁棒归一化后，逐列分位阈值 + 每行 top-N，
      产出 subtype=cell 候选（类别/缺失/拼写）。
    - GANomaly 走"行级"通道：行异常分超过干净分位阈值即判为可疑行，行内按归一化逐列
      贡献取 top-k 列粗定位，产出 subtype=row 候选（多列联合异常）。
    - 两路候选按 (row_id, column) 融合（cell 优先），error_type 统一为 DIST。

输出 schema:
    row_id, column, value, error_type(=DIST), anomaly_score, col_contribution, subtype
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from stage_2.encoding import TabularEncoder
from stage_2.model import BaseAnomalyModel


def column_contributions(
    model: BaseAnomalyModel,
    encoder: TabularEncoder,
    x: np.ndarray,
) -> pd.DataFrame:
    """每行每列的重构误差贡献（逐列求和的原始误差）。"""
    per_feature = model.per_feature_error(x)
    col_errors = encoder.aggregate_feature_errors(per_feature)
    return pd.DataFrame(col_errors)


def robust_normalize(contrib: pd.DataFrame, row_clean: np.ndarray) -> pd.DataFrame:
    """
    按列鲁棒归一化（中位数/MAD），消除宽类别列与数值列之间的量纲差异，
    使逐列贡献可跨列比较（直接影响每行 top-N 选择的正确性）。
    """
    out = pd.DataFrame(index=contrib.index)
    for col in contrib.columns:
        v = contrib[col].to_numpy(dtype=np.float64)
        clean_v = v[row_clean]
        med = float(np.median(clean_v)) if len(clean_v) else float(np.median(v))
        mad = float(np.median(np.abs(clean_v - med))) or 1.0
        out[col] = (v - med) / (1.4826 * mad)
    return out


def column_thresholds(
    clean_contrib: pd.DataFrame,
    quantile: float = 0.99,
) -> dict[str, float]:
    """基于干净子集的逐列（归一化）误差分布，取高分位作为每列阈值。"""
    return {col: float(clean_contrib[col].quantile(quantile)) for col in clean_contrib.columns}


def flag_suspicious_cells(
    df: pd.DataFrame,
    contrib: pd.DataFrame,
    thresholds: dict[str, float],
    row_scores: Optional[np.ndarray] = None,
    max_cells_per_row: int = 0,
    subtype: str = "cell",
) -> pd.DataFrame:
    """依据逐列阈值标记可疑单元格（DAE 单元格级通道）。"""
    records: list[dict] = []
    for col in contrib.columns:
        thr = thresholds.get(col)
        if thr is None:
            continue
        col_err = contrib[col].to_numpy()
        for pos, err in enumerate(col_err):
            if err <= thr:
                continue
            records.append({
                "row_id": df.index[pos],
                "column": col,
                "value": df.iloc[pos][col],
                "error_type": "DIST",
                "anomaly_score": float(row_scores[pos]) if row_scores is not None else float(err),
                "col_contribution": float(err),
                "subtype": subtype,
            })
    result = pd.DataFrame(records)
    if max_cells_per_row and not result.empty:
        result = (
            result.sort_values("col_contribution", ascending=False)
            .groupby("row_id", group_keys=False)
            .head(max_cells_per_row)
            .reset_index(drop=True)
        )
    return result


def flag_suspicious_rows(
    df: pd.DataFrame,
    contrib_norm: pd.DataFrame,
    row_scores: np.ndarray,
    row_clean: np.ndarray,
    row_quantile: float = 0.99,
    top_k_cols: int = 3,
    min_col_z: float = 0.0,
    subtype: str = "row",
) -> pd.DataFrame:
    """
    行级异常通道（GANomaly）：行异常分超过干净分位阈值的行判为可疑，
    行内按归一化逐列贡献取 top-k 列做粗定位。
    """
    clean_scores = row_scores[row_clean]
    thr = float(np.quantile(clean_scores, row_quantile)) if len(clean_scores) else float("inf")
    records: list[dict] = []
    cols = list(contrib_norm.columns)
    for pos in np.where(row_scores > thr)[0]:
        row_vals = contrib_norm.iloc[pos]
        ranked = row_vals.sort_values(ascending=False)
        for col in ranked.index[:top_k_cols]:
            if float(ranked[col]) < min_col_z:
                continue
            records.append({
                "row_id": df.index[pos],
                "column": col,
                "value": df.iloc[pos][col],
                "error_type": "DIST",
                "anomaly_score": float(row_scores[pos]),
                "col_contribution": float(ranked[col]),
                "subtype": subtype,
            })
    return pd.DataFrame(records)


def _fuse(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """按 (row_id, column) 融合多路候选，cell 优先（列表中靠前者优先）。"""
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=[
            "row_id", "column", "value", "error_type",
            "anomaly_score", "col_contribution", "subtype",
        ])
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["row_id", "column"], keep="first").reset_index(drop=True)
    return combined


def run_stage2(
    df: pd.DataFrame,
    clean_mask: pd.DataFrame,
    *,
    dae: Optional[BaseAnomalyModel] = None,
    ganomaly: Optional[BaseAnomalyModel] = None,
    mode: str = "fuse",
    max_cardinality: int = 50,
    n_hash: int = 16,
    quantile: float = 0.995,
    max_cells_per_row: int = 1,
    row_quantile: float = 0.99,
    row_top_k: int = 3,
    row_min_col_z: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Stage 2 端到端：编码 -> 仅干净行训练 -> 全量打分 -> 融合候选。

    Returns:
        (combined_candidates, {"dae": ..., "ganomaly": ...})
    """
    encoder = TabularEncoder(max_cardinality=max_cardinality, n_hash=n_hash)
    encoder.fit(df, clean_mask=clean_mask)
    spec = encoder.feature_spec()

    row_clean = clean_mask.all(axis=1).to_numpy()
    x_all = encoder.transform(df)
    x_clean = x_all[row_clean]

    parts: dict[str, pd.DataFrame] = {}

    if mode in ("dae", "fuse") and dae is not None:
        dae.fit(x_clean, feature_spec=spec)
        contrib = column_contributions(dae, encoder, x_all)
        contrib_norm = robust_normalize(contrib, row_clean)
        thresholds = column_thresholds(contrib_norm[row_clean], quantile=quantile)
        row_scores = dae.anomaly_score(x_all)
        parts["dae"] = flag_suspicious_cells(
            df, contrib_norm, thresholds,
            row_scores=row_scores, max_cells_per_row=max_cells_per_row, subtype="cell",
        )

    if mode in ("ganomaly", "fuse") and ganomaly is not None:
        ganomaly.fit(x_clean, feature_spec=spec)
        contrib = column_contributions(ganomaly, encoder, x_all)
        contrib_norm = robust_normalize(contrib, row_clean)
        row_scores = ganomaly.anomaly_score(x_all)
        parts["ganomaly"] = flag_suspicious_rows(
            df, contrib_norm, row_scores, row_clean,
            row_quantile=row_quantile, top_k_cols=row_top_k,
            min_col_z=row_min_col_z, subtype="row",
        )

    # 融合：cell 优先于 row
    combined = _fuse([parts.get("dae"), parts.get("ganomaly")])
    return combined, parts
