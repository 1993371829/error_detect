"""
检测器四：近似函数依赖异常（文档 §11）。

FD 检测已从 Stage 1 迁移至此，作为 Stage 2 多检测器之一。默认开启 LLM 语义校验
（semantic_check=True），过滤掉非语义依赖的伪 FD；无 LLM 时自动退化为纯统计高召回。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from stage_1.fd_detect import detect_vad
from stage_2.detectors.base import DetectorContext
from stage_2.schema import CandidateError


def detect_fd(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    min_confidence: float = 0.9,
    min_group_support: int = 30,
    min_group_confidence: float = 0.97,
    pure_threshold: float = 0.9,
    min_pure_group_ratio: float = 0.8,
    semantic_check: bool = True,
    llm=None,
) -> list[CandidateError]:
    vad_errors, _ = detect_vad(
        df,
        min_confidence=min_confidence,
        min_group_support=min_group_support,
        min_group_confidence=min_group_confidence,
        pure_threshold=pure_threshold,
        min_pure_group_ratio=min_pure_group_ratio,
        llm=llm,
        semantic_check=semantic_check and llm is not None,
    )
    cands: list[CandidateError] = []
    for e in vad_errors:
        cands.append(CandidateError(
            row_id=int(e["row_id"]), column=str(e["column"]), value=e.get("value"),
            detector="approx_fd", error_type="VAD",
            score=float(e.get("confidence", 0.9) or 0.9),
            evidence=str(e.get("reason", "")),
            suggested_fix=e.get("suggested_fix"),
            metadata={"violated_rule": e.get("violated_rule")},
        ))
    return cands
