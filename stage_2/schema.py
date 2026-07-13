"""
Stage 2 多检测器框架的统一数据结构（借鉴文档 §3.3 / §3.4）。

所有检测器（统计/低频拼写/近邻一致性/形态离群/数值格式/自监督重构）
都输出统一的 CandidateError；跨检测器融合见 stage_2/fusion.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class CandidateError:
    """单个检测器对某单元格的一条候选判定。"""

    row_id: int
    column: str
    value: Any
    detector: str                       # 检测器名（statistical/categorical/...）
    error_type: str                     # DIST/VAD/FI/...（融合前的初判类型）
    score: float                        # 该检测器原始异常分（越大越可疑）
    evidence: str = ""                  # 人类可读证据说明
    suggested_fix: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "row_id": int(self.row_id),
            "column": str(self.column),
            "value": "" if self.value is None else str(self.value),
            "detector": self.detector,
            "error_type": self.error_type,
            "anomaly_score": float(self.score),
            "evidence": self.evidence,
            "suggested_fix": "" if self.suggested_fix is None else str(self.suggested_fix),
            "subtype": str(self.metadata.get("subtype", "")),
        }


def candidates_to_frame(cands: list[CandidateError]) -> pd.DataFrame:
    """CandidateError 列表 -> DataFrame（统一 schema）。"""
    cols = ["row_id", "column", "value", "detector", "error_type",
            "anomaly_score", "evidence", "suggested_fix", "subtype"]
    if not cands:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([c.to_record() for c in cands], columns=cols)
