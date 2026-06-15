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
    合并 Stage1 错误与 Stage2 DIST 候选，按 (row_id, column) 去重，Stage1 优先。

    返回列为两者字段的并集（缺失填空），并带 source 标记。
    """
    s1 = stage1_errors.copy()
    s2 = stage2_candidates.copy()
    if not s1.empty:
        s1["source"] = "stage1"
        s1_cells = set(zip(s1["row_id"], s1["column"]))
    else:
        s1_cells = set()
    if not s2.empty:
        s2["source"] = "stage2"
        s2 = s2[~s2.apply(lambda r: (r["row_id"], r["column"]) in s1_cells, axis=1)]

    combined = pd.concat([s1, s2], ignore_index=True, sort=False)
    return combined
