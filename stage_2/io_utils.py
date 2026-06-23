"""
Stage 2 输入输出辅助：表格/掩码读取与候选合并，供 cli 与 evaluate 复用。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 与 Stage 1 一致的 CSV 读取策略（避免空串被转成 NaN）
READ_KWARGS = dict(dtype=str, keep_default_na=False, na_values=[""])


def read_table(path: str | Path) -> pd.DataFrame:
    """按 Stage 1 同款策略读取表格 CSV。"""
    return pd.read_csv(path, **READ_KWARGS)


def read_clean_mask(path: str | Path) -> pd.DataFrame:
    """
    读取 Stage 1 导出的干净单元格掩码并稳健地还原为 bool。

    直接 astype(bool) 会把字符串 'False' 误判为 True，故显式映射。
    """
    raw = pd.read_csv(path)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    out = {}
    for col in raw.columns:
        s = raw[col]
        if s.dtype == bool:
            out[col] = s
        else:
            out[col] = s.astype(str).str.strip().str.lower().map(mapping).fillna(True)
    return pd.DataFrame(out)


def merge_candidates(
    stage1_errors: pd.DataFrame,
    stage2_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    多证据融合合并 Stage1 错误与 Stage2 候选（文档 §15）。

    按 (row_id, column) 聚合所有检测器证据，加权融合出 suspicion_score 与
    confidence_tier，并保留 Stage3 所需原字段（error_type/violated_rule/reason/
    suggested_fix/anomaly_score/subtype/source）。
    """
    from stage_2.fusion import fuse_candidates
    return fuse_candidates(stage1_errors, stage2_candidates)
