"""
Stage 3 上下文构造：把候选错误按行分组，并附带整行值、同列正常样例、列语义类型，
供 prompt 构造使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.io_utils import read_table


@dataclass
class SuspectCell:
    """一个可疑单元格及其来自前序阶段的线索。"""

    column: str
    value: str
    prior_error_type: str            # 前序阶段判定的类型（FI/MV/T/VAD/DIST）
    prior_source: str                # stage1 / stage2
    reason: str                      # 前序原因说明
    suggested_fix: str               # 前序建议修复（可能为空）
    semantic_type: str               # 该列语义类型（来自 rules.json）
    normal_samples: list = field(default_factory=list)  # 该列正常高频样例


@dataclass
class RowContext:
    """一行的完整上下文：整行值 + 该行所有可疑单元格。"""

    row_id: int
    row_values: dict                 # 列 -> 值（整行，用于跨列判断）
    suspects: list                   # list[SuspectCell]


def load_semantic_types(rules_path: str | Path) -> dict:
    """从 Stage1 rules.json 读取 列 -> semantic_type 映射（跳过 FD 汇总条目）。"""
    path = Path(rules_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    out = {}
    for entry in rules:
        if isinstance(entry, dict) and "column" in entry:
            out[entry["column"]] = entry.get("semantic_type", "")
    return out


def compute_normal_samples(df: pd.DataFrame, max_samples: int = 8) -> dict:
    """每列取出现频次最高的若干非空值作为"正常样例"。"""
    out = {}
    for col in df.columns:
        series = df[col]
        non_blank = series[~series.map(is_blank)].astype(str)
        if non_blank.empty:
            out[col] = []
        else:
            out[col] = non_blank.value_counts().head(max_samples).index.tolist()
    return out


def build_row_contexts(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    semantic_types: dict,
    normal_samples: dict,
) -> list[RowContext]:
    """按 row_id 分组候选，组装 RowContext 列表（按 row_id 升序）。"""
    contexts: list[RowContext] = []
    candidates = candidates.copy()
    candidates["row_id"] = candidates["row_id"].astype(int)

    for row_id, group in candidates.groupby("row_id", sort=True):
        if row_id < 0 or row_id >= len(df):
            continue
        row_values = {col: df.iloc[row_id][col] for col in df.columns}
        suspects = []
        for _, r in group.iterrows():
            col = str(r["column"])
            suspects.append(SuspectCell(
                column=col,
                value="" if pd.isna(r.get("value")) else str(r.get("value", "")),
                prior_error_type=str(r.get("error_type", "") or ""),
                prior_source=str(r.get("source", "") or ""),
                reason=str(r.get("reason", "") or ""),
                suggested_fix="" if pd.isna(r.get("suggested_fix")) else str(r.get("suggested_fix", "") or ""),
                semantic_type=semantic_types.get(col, ""),
                normal_samples=normal_samples.get(col, []),
            ))
        contexts.append(RowContext(row_id=int(row_id), row_values=row_values, suspects=suspects))
    return contexts


def load_contexts(
    input_csv: str | Path,
    candidates_csv: str | Path,
    rules_json: str | Path,
    max_normal_samples: int = 8,
) -> tuple[pd.DataFrame, list[RowContext]]:
    """一站式加载：返回 (原始表, RowContext 列表)。"""
    df = read_table(input_csv)
    candidates = read_table(candidates_csv)
    semantic_types = load_semantic_types(rules_json)
    normal_samples = compute_normal_samples(df, max_samples=max_normal_samples)
    contexts = build_row_contexts(df, candidates, semantic_types, normal_samples)
    return df, contexts
