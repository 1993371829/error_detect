"""
检测器六：自监督重构异常（文档 §13）。

包装现有 ConditionalPredictor（masked-column 条件预测）的打分结果，
转换为统一 CandidateError。模型训练/打分仍由 stage_2.score 负责，
本模块只做「DataFrame -> CandidateError」适配，保留已验证逻辑。
"""

from __future__ import annotations

import pandas as pd

from stage_2.schema import CandidateError


def recon_frame_to_candidates(recon: pd.DataFrame) -> list[CandidateError]:
    """flag_suspicious_cells 的输出 DataFrame -> list[CandidateError]。"""
    if recon is None or recon.empty:
        return []
    cands: list[CandidateError] = []
    for _, r in recon.iterrows():
        score = float(r.get("anomaly_score", 0.0) or 0.0)
        fix = r.get("suggested_fix")
        fix = None if (fix is None or (isinstance(fix, float) and pd.isna(fix)) or fix == "") else fix
        cands.append(CandidateError(
            row_id=int(r["row_id"]), column=str(r["column"]), value=r.get("value"),
            detector="reconstruction", error_type=str(r.get("error_type", "DIST") or "DIST"),
            score=score,
            evidence=f"条件预测异常分 {score:.3g}（子类 {r.get('subtype', '')}）",
            suggested_fix=fix,
            metadata={"subtype": str(r.get("subtype", "") or "")},
        ))
    return cands
