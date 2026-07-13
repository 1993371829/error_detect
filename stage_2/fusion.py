"""
多证据加权融合与置信度分层（文档 §15）。

把 Stage1 规则证据与 Stage2 各检测器证据按 (row_id, column) 聚合：
    1. 每个检测器分数归一化到 [0,1]（软检测器用分位归一化，硬/规则检测器用其置信度）。
    2. 加权融合 S = 1 - ∏(1 - w_d * s_d)（多检测器一致时分数自然升高）。
    3. 分层 high(>=0.85)/mid(>=0.6)/low(>=0.4)（<0.4 记为 low，不丢弃以保召回）。
    4. 输出每格 suspicion_score / confidence_tier / evidence / candidate_fixes，
       并保留 Stage1/Stage2 原字段，确保 Stage 3 向后兼容。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 软检测器：原始分是 z-score / -logP 等无界量纲，用分位归一化到 [0,1]
_SOFT_DETECTORS = {"reconstruction", "statistical", "neighbor_consistency"}

# 检测器权重（文档 §15.3）
_WEIGHTS = {
    "strong_rule": 1.0,
    "approx_fd": 0.9,
    "approx_fd_medium": 0.5,   # 双轨 medium 级 FD：仅作弱证据（不进 mask，融合权重降低）
    "reconstruction": 0.85,
    "categorical_typo": 0.8,
    "typo": 0.8,
    "neighbor_consistency": 0.75,
    "pattern_outlier": 0.6,
    "numeric_format": 0.85,   # 数值元数违规：规则性强、高精度
    "statistical": 0.6,
    "format_cluster": 0.6,
    "value_burst": 0.6,      # 异常高频重复值：低权重进融合，交 Stage 3 裁决
    "secondary_format": 0.5,  # 次级日期格式簇：低权重，交 Stage 3 裁决
}
_DEFAULT_WEIGHT = 0.6

# 检测器 -> 证据"族"。独立族数（family_count）衡量多源一致程度：跨族命中越多，
# 越可能是真错（供 Stage 3 优先保留，缓解精检后召回下降）。同族内多个检测器高度相关，
# 只算 1 次，避免"同质证据堆叠"虚高。
_DETECTOR_FAMILY = {
    "strong_rule": "rule",
    "approx_fd": "rule",
    "approx_fd_medium": "rule",
    "reconstruction": "model",
    "statistical": "statistical",
    "neighbor_consistency": "neighbor",
    "pattern_outlier": "pattern",
    "format_cluster": "pattern",
    "secondary_format": "pattern",
    "numeric_format": "format",
    "typo": "text",
    "categorical_typo": "text",
    "value_burst": "burst",
}
_DEFAULT_FAMILY = "other"


def _family(detector: str) -> str:
    return _DETECTOR_FAMILY.get(detector, _DEFAULT_FAMILY)


_TIER_HIGH = 0.85
_TIER_MID = 0.60
_TIER_LOW = 0.40


def _stage1_detector(row: pd.Series) -> str:
    """Stage1 错误 -> 统一 detector 名（供权重/证据展示）。"""
    vr = str(row.get("violated_rule", "") or "")
    et = str(row.get("error_type", "") or "").upper()
    sev = str(row.get("severity", "high") or "high").strip().lower()
    if et in ("T",):
        return "typo"
    if vr.startswith("fd:") or et == "VAD":
        # 双轨：medium 级 FD 仅作弱证据（降低融合权重）
        return "approx_fd_medium" if sev == "medium" else "approx_fd"
    return "strong_rule"


def _weight(detector: str) -> float:
    return _WEIGHTS.get(detector, _DEFAULT_WEIGHT)


def _normalize_scores(records: list[dict]) -> None:
    """就地为每条记录写入 s_norm（[0,1]）。软检测器按 detector 分位归一化。"""
    by_det: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_det.setdefault(r["detector"], []).append(i)
    for det, idxs in by_det.items():
        if det in _SOFT_DETECTORS:
            scores = np.array([records[i]["score"] for i in idxs], dtype=float)
            order = scores.argsort()
            ranks = np.empty(len(scores))
            # 分位排名：最大值 ->1，最小值 -> 1/n（至少给一点权重）
            ranks[order] = (np.arange(len(scores)) + 1) / len(scores)
            for j, i in enumerate(idxs):
                records[i]["s_norm"] = float(ranks[j])
        else:
            for i in idxs:
                s = records[i]["score"]
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    s = 0.9
                if math.isnan(s) or s <= 0.0:
                    s = 0.9
                records[i]["s_norm"] = float(min(max(s, 0.0), 1.0))


def _collect_records(
    stage1_errors: pd.DataFrame,
    stage2_candidates: pd.DataFrame,
) -> list[dict]:
    """把两阶段证据规整为统一记录列表。"""
    records: list[dict] = []
    if stage1_errors is not None and not stage1_errors.empty:
        for _, r in stage1_errors.iterrows():
            det = _stage1_detector(r)
            conf = r.get("confidence")
            try:
                score = float(conf)
                if math.isnan(score):
                    score = 0.9
            except (TypeError, ValueError):
                score = 0.9
            records.append({
                "row_id": int(r["row_id"]), "column": str(r["column"]),
                "value": r.get("value"), "detector": det, "source": "stage1",
                "error_type": str(r.get("error_type", "") or ""),
                "violated_rule": r.get("violated_rule"),
                "reason": r.get("reason", ""), "suggested_fix": r.get("suggested_fix"),
                "anomaly_score": None, "subtype": "", "score": score,
            })
    if stage2_candidates is not None and not stage2_candidates.empty:
        for _, r in stage2_candidates.iterrows():
            det = str(r.get("detector", "reconstruction") or "reconstruction")
            records.append({
                "row_id": int(r["row_id"]), "column": str(r["column"]),
                "value": r.get("value"), "detector": det, "source": "stage2",
                "error_type": str(r.get("error_type", "DIST") or "DIST"),
                "violated_rule": det,
                "reason": r.get("evidence", ""), "suggested_fix": r.get("suggested_fix"),
                "anomaly_score": r.get("anomaly_score"),
                "subtype": r.get("subtype", ""),
                "score": r.get("anomaly_score", 0.0),
            })
    return records


def _tier(score: float) -> str:
    if score >= _TIER_HIGH:
        return "high"
    if score >= _TIER_MID:
        return "mid"
    return "low"


def fuse_candidates(
    stage1_errors: pd.DataFrame,
    stage2_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    融合 Stage1 + Stage2 证据，按 (row_id, column) 输出一行，含 suspicion_score /
    confidence_tier / evidence / candidate_fixes，并保留 Stage3 所需原字段。
    """
    records = _collect_records(stage1_errors, stage2_candidates)
    if not records:
        return pd.DataFrame(columns=[
            "row_id", "column", "value", "error_type", "violated_rule", "reason",
            "suggested_fix", "confidence", "anomaly_score", "subtype", "source",
            "suspicion_score", "confidence_tier", "evidence", "candidate_fixes", "detectors",
            "families", "family_count",
        ])
    _normalize_scores(records)

    cells: dict[tuple, list[dict]] = {}
    for r in records:
        cells.setdefault((r["row_id"], r["column"]), []).append(r)

    out_rows: list[dict] = []
    for (row_id, col), evs in cells.items():
        # 每检测器取最强证据
        best_by_det: dict[str, dict] = {}
        for e in evs:
            d = e["detector"]
            if d not in best_by_det or e["s_norm"] > best_by_det[d]["s_norm"]:
                best_by_det[d] = e
        prod = 1.0
        for d, e in best_by_det.items():
            prod *= (1.0 - _weight(d) * e["s_norm"])
        suspicion = 1.0 - prod

        has_s1 = [e for e in evs if e["source"] == "stage1"]
        if has_s1:
            primary = max(has_s1, key=lambda e: _weight(e["detector"]) * e["s_norm"])
            source = "stage1"
        else:
            primary = max(evs, key=lambda e: _weight(e["detector"]) * e["s_norm"])
            source = "stage2"

        # anomaly_score 优先取 reconstruction（Stage3 共识保护用），否则取最大软分
        recon = best_by_det.get("reconstruction")
        anomaly = recon["anomaly_score"] if recon is not None else primary.get("anomaly_score")
        subtype = recon["subtype"] if recon is not None else primary.get("subtype", "")

        fixes = []
        for e in evs:
            f = e.get("suggested_fix")
            if f not in (None, "", "nan") and not (isinstance(f, float) and pd.isna(f)):
                if str(f) not in fixes:
                    fixes.append(str(f))
        evidence_text = " | ".join(
            f"[{e['detector']}] {str(e.get('reason') or '')}".strip() for e in evs
        )
        detectors = ",".join(sorted(best_by_det.keys()))
        families = sorted({_family(d) for d in best_by_det})
        family_count = len(families)

        out_rows.append({
            "row_id": int(row_id), "column": col,
            "value": primary.get("value"),
            "error_type": primary.get("error_type", "DIST"),
            "violated_rule": primary.get("violated_rule"),
            "reason": primary.get("reason", ""),
            "suggested_fix": primary.get("suggested_fix"),
            "confidence": round(float(suspicion), 3),
            "anomaly_score": anomaly,
            "subtype": subtype,
            "source": source,
            "suspicion_score": round(float(suspicion), 4),
            "confidence_tier": _tier(suspicion),
            "evidence": evidence_text,
            "candidate_fixes": "; ".join(fixes),
            "detectors": detectors,
            "families": ",".join(families),
            "family_count": family_count,
        })
    out = pd.DataFrame(out_rows)
    out = out.sort_values(["row_id", "column"]).reset_index(drop=True)
    return out
