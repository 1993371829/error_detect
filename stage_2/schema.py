"""
Stage 2 多检测器框架的统一数据结构（借鉴文档 §3.3 / §3.4）。

所有检测器（统计/低频拼写/关联规则/近似FD/近邻一致性/自监督重构/聚类）
都输出统一的 CandidateError；证据融合后聚合为 SuspiciousCell。
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


@dataclass
class SuspiciousCell:
    """同一单元格多检测器证据融合后的结果（文档 §3.4）。"""

    row_id: int
    column: str
    value: Any
    suspicion_score: float
    confidence_tier: str                # high / mid / low
    evidence_list: list = field(default_factory=list)   # list[CandidateError]
    candidate_fixes: list = field(default_factory=list)
    error_type: str = "DIST"
    source: str = "stage2"

    def evidence_text(self) -> str:
        parts = []
        for e in self.evidence_list:
            msg = e.evidence or f"{e.detector} score={e.score:.3g}"
            parts.append(f"[{e.detector}] {msg}")
        return " | ".join(parts)


def candidates_to_frame(cands: list[CandidateError]) -> pd.DataFrame:
    """CandidateError 列表 -> DataFrame（统一 schema）。"""
    cols = ["row_id", "column", "value", "detector", "error_type",
            "anomaly_score", "evidence", "suggested_fix", "subtype"]
    if not cands:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([c.to_record() for c in cands], columns=cols)
