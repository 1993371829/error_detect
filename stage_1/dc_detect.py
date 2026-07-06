"""
否定约束(DC)检测（二期）——仅覆盖 FD/CFD 结构上无法表达的**非等值谓词**。

为避免与 FD/CFD（等值依赖）冗余，本模块**不做通用 DC 发现**，只系统化以下几类：
    1. arith   : LLM 归纳的跨列算术约束（复用 stage_1/arith_rules）。      -> FI
    2. compare : 跨列大小比较（start<=end / min<=max / low<=high 等）。      -> FI
    3. temporal: 年份/时间落在合理区间。                                    -> FI
    4. order   : 排序单调一致性（rank 与度量列反序/同序一致）。            -> VAD

compare/temporal/order 为确定性结构约束，走统计验证（高支持 + 低违反 -> high），
不额外消耗 LLM；arith 沿用其自身验证并记为 high。空值不参与。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.profiling import is_blank
from stage_2.encoding import _extract_number


# 低值列关键词 -> 对应高值列关键词（约束: 低 <= 高）
_COMPARE_KEYWORDS = [
    ("start", "end"), ("begin", "end"), ("from", "to"), ("min", "max"),
    ("low", "high"), ("open", "close"), ("first", "last"),
    ("earliest", "latest"), ("lower", "upper"), ("before", "after"),
]


def _num_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """列转数值数组（不可解析 -> np.nan）。"""
    out = np.full(len(df), np.nan)
    for i, v in enumerate(df[col].to_numpy()):
        if is_blank(v):
            continue
        n = _extract_number(str(v))
        if n is not None and not (isinstance(n, float) and np.isnan(n)):
            out[i] = float(n)
    return out


def _normalize_base(col: str, keyword: str) -> str:
    """去掉关键词与非字母数字后的列名主干，用于配对（start_year / end_year -> year）。"""
    s = str(col).lower()
    s = s.replace(keyword, "")
    return "".join(ch for ch in s if ch.isalnum())


def _grade_structural(
    support: int, confidence: float, n_rows: int, thresholds,
    recompute_confidence=None,
) -> str:
    """确定性结构约束分档（不调 LLM，灰区保守 medium）。"""
    from stage_1.rule_validation import grade_rule
    severity, _stab, _reason = grade_rule(
        support=support, confidence=confidence, n_rows=n_rows, th=thresholds,
        recompute_confidence=recompute_confidence, llm=None, cache=None,
        audit_prompt=None,
    )
    return severity


def detect_compare(df: pd.DataFrame, *, cfg, thresholds) -> list[dict]:
    """跨列大小比较：按列名配对，约束 lo<=hi，标记违反行的两端单元格。"""
    cols = [str(c) for c in df.columns]
    errors: list[dict] = []
    seen_pairs: set = set()
    for lo_kw, hi_kw in _COMPARE_KEYWORDS:
        lo_cols = [c for c in cols if lo_kw in c.lower()]
        hi_cols = [c for c in cols if hi_kw in c.lower()]
        for lo in lo_cols:
            for hi in hi_cols:
                if lo == hi:
                    continue
                if _normalize_base(lo, lo_kw) != _normalize_base(hi, hi_kw):
                    continue
                key = tuple(sorted((lo, hi)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                a, b = _num_series(df, lo), _num_series(df, hi)
                comparable = ~(np.isnan(a) | np.isnan(b))
                support = int(comparable.sum())
                if support < cfg.min_support:
                    continue
                satisfied = int((a[comparable] <= b[comparable]).sum())
                confidence = satisfied / support
                if 1.0 - confidence > cfg.max_violation_rate:
                    continue  # 约束本身不成立，丢弃
                sev = _grade_structural(support, confidence, len(df), thresholds)
                if sev == "drop":
                    continue
                viol = comparable & (a > b)
                for pos in np.where(viol)[0]:
                    real_row_id = df.index[pos]
                    for col in (lo, hi):
                        errors.append({
                            "row_id": real_row_id, "column": col, "value": df.iloc[pos][col],
                            "error_type": "FI",
                            "violated_rule": f"dc:compare:{lo}<={hi}",
                            "reason": f"违反跨列比较约束 {lo}<={hi}: {lo}={a[pos]:g}, {hi}={b[pos]:g}",
                            "confidence": round(confidence, 3),
                            "severity": sev,
                        })
    return errors


def detect_temporal(df: pd.DataFrame, *, cfg, thresholds) -> list[dict]:
    """年份合理性：识别年份列，标记落在 [1800, 当前年+1] 之外的值。"""
    cur_year = datetime.now().year
    lo_year, hi_year = 1800, cur_year + 1
    errors: list[dict] = []
    for col in df.columns:
        name = str(col).lower()
        vals = _num_series(df, col)
        present = ~np.isnan(vals)
        support = int(present.sum())
        if support < cfg.min_support:
            continue
        v = vals[present]
        # 年份列判据：列名含 year，或绝大多数值是 [1000,2100] 的 4 位整数
        name_hint = "year" in name or name.endswith("yr")
        int_like = np.mean(np.isclose(v, np.round(v))) > 0.95
        in_year_range = np.mean((v >= 1000) & (v <= 2100)) > 0.9
        if not (name_hint or (int_like and in_year_range)):
            continue
        in_ok = (v >= lo_year) & (v <= hi_year)
        confidence = float(in_ok.mean())
        if 1.0 - confidence > cfg.max_violation_rate:
            continue
        sev = _grade_structural(support, confidence, len(df), thresholds)
        if sev == "drop":
            continue
        viol = present & ((vals < lo_year) | (vals > hi_year))
        for pos in np.where(viol)[0]:
            errors.append({
                "row_id": df.index[pos], "column": str(col), "value": df.iloc[pos][col],
                "error_type": "FI",
                "violated_rule": "dc:temporal:year_range",
                "reason": f"年份 {vals[pos]:g} 超出合理区间 [{lo_year}, {hi_year}]",
                "confidence": round(confidence, 3),
                "severity": sev,
            })
    return errors


def detect_order(df: pd.DataFrame, *, cfg, thresholds) -> list[dict]:
    """
    排序单调一致性：rank 列（近似整数全排列）与数值度量列应单调同/反序。

    按 rank 升序排列后，度量列应单调；对同时与前驱、后继都逆序的行（局部单点异常）标记度量列。
    """
    errors: list[dict] = []
    n = len(df)
    if n < max(cfg.min_support, 5):
        return errors
    # 识别 rank 列：近似唯一的整数序列
    rank_cols = []
    for col in df.columns:
        vals = _num_series(df, col)
        present = ~np.isnan(vals)
        if int(present.sum()) < cfg.min_support:
            continue
        v = vals[present]
        if np.mean(np.isclose(v, np.round(v))) > 0.95 and (len(np.unique(v)) / len(v)) > 0.95:
            rank_cols.append(col)
    if not rank_cols:
        return errors

    for rank_col in rank_cols:
        r = _num_series(df, rank_col)
        for meas_col in df.columns:
            if meas_col == rank_col:
                continue
            m = _num_series(df, meas_col)
            valid = ~(np.isnan(r) | np.isnan(m))
            support = int(valid.sum())
            if support < cfg.min_support:
                continue
            if len(np.unique(m[valid])) < 3:
                continue
            order = np.argsort(r[valid], kind="stable")
            m_sorted = m[valid][order]
            pos_map = np.where(valid)[0][order]
            up = int(np.sum(np.diff(m_sorted) > 0))
            down = int(np.sum(np.diff(m_sorted) < 0))
            total_adj = up + down
            if total_adj < cfg.min_support:
                continue
            direction = 1 if up >= down else -1  # 主导方向：+1 同序, -1 反序
            consistent = max(up, down)
            confidence = consistent / total_adj
            if 1.0 - confidence > cfg.max_violation_rate:
                continue
            sev = _grade_structural(support, confidence, len(df), thresholds)
            if sev == "drop":
                continue
            # 局部单点异常：在应单调的序列中形成"局部极值"（尖峰/凹陷），且幅度显著
            # 超过典型相邻步长（幅度门槛可抑制被尖峰带偏的邻点）。
            adj = np.abs(np.diff(m_sorted))
            scale = float(np.median(adj[adj > 0])) if np.any(adj > 0) else 0.0
            thresh = scale * 3.0
            for k in range(1, len(m_sorted) - 1):
                prev, cur, nxt = m_sorted[k - 1], m_sorted[k], m_sorted[k + 1]
                is_max = cur > prev and cur > nxt
                is_min = cur < prev and cur < nxt
                if not (is_max or is_min):
                    continue
                dev = (cur - max(prev, nxt)) if is_max else (min(prev, nxt) - cur)
                if scale > 0 and dev < thresh:
                    continue
                pos = pos_map[k]
                errors.append({
                    "row_id": df.index[pos], "column": str(meas_col),
                    "value": df.iloc[pos][meas_col],
                    "error_type": "VAD",
                    "violated_rule": f"dc:order:{rank_col}~{meas_col}",
                    "reason": (
                        f"违反排序单调一致性: 按 {rank_col} 排序后 {meas_col} 应"
                        f"{'同序递增' if direction > 0 else '反序递减'}，此点为显著局部极值"
                    ),
                    "confidence": round(confidence, 3),
                    "severity": sev,
                })
    return errors


def detect_dc(
    df: pd.DataFrame,
    *,
    cfg,
    thresholds,
    llm=None,
) -> list[dict]:
    """DC 总入口：arith(LLM) + compare + temporal + order。返回带 severity 的错误列表。"""
    errors: list[dict] = []

    if cfg.enable_arith and llm is not None:
        from stage_1.arith_rules import generate_arithmetic_rules, validate_and_apply
        rules = generate_arithmetic_rules(llm, df)
        if rules:
            for err in validate_and_apply(
                df, rules, min_support=cfg.min_support, max_violation_rate=cfg.max_violation_rate,
            ):
                err["severity"] = "high"  # 算术约束确定性强
                err["violated_rule"] = "dc:arith"
                errors.append(err)

    if cfg.enable_compare:
        errors.extend(detect_compare(df, cfg=cfg, thresholds=thresholds))
    if cfg.enable_temporal:
        errors.extend(detect_temporal(df, cfg=cfg, thresholds=thresholds))
    if cfg.enable_order:
        errors.extend(detect_order(df, cfg=cfg, thresholds=thresholds))

    print(f"否定约束检测: 共 {len(errors)} 个候选错误 (DC: arith/compare/temporal/order)")
    return errors
