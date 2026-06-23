"""
检测器三：关联规则异常（文档 §10）。

在类别列上挖掘单前件关联规则 (A=a) => (B=b)，保留 support/confidence/lift 达标的
高可信规则，标记违反 consequent 的单元格。适合类别组合不一致（country->currency 等）。
为避免组合爆炸，仅枚举单前件、低基数类别列。默认关闭（与 FD 有重叠，按需开启）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.detectors.base import DetectorContext
from stage_2.schema import CandidateError


def detect_association(
    df: pd.DataFrame,
    clean_mask: Optional[pd.DataFrame],
    ctx: DetectorContext,
    *,
    min_support: int = 30,
    min_confidence: float = 0.98,
    min_lift: float = 2.0,
    max_cardinality: int = 100,
) -> list[CandidateError]:
    n = len(df)
    cat_cols = [
        str(c) for c in df.columns
        if ctx.kinds.get(str(c)) == "categorical"
        and 1 < int(df[c][~df[c].map(is_blank)].astype(str).nunique()) <= max_cardinality
    ]
    if len(cat_cols) < 2:
        return []

    # B 列各取值全局占比（计算 lift）
    global_share: dict[str, dict[str, float]] = {}
    for b in cat_cols:
        nb = df[b][~df[b].map(is_blank)].astype(str)
        vc = nb.value_counts(normalize=True)
        global_share[b] = {str(k): float(v) for k, v in vc.items()}

    cands: list[CandidateError] = []
    for a in cat_cols:
        for b in cat_cols:
            if a == b:
                continue
            groups: dict[str, "defaultdict[str, int]"] = defaultdict(lambda: defaultdict(int))
            for av, bv in zip(df[a], df[b]):
                if is_blank(av) or is_blank(bv):
                    continue
                groups[str(av)][str(bv)] += 1
            # 每个达标规则的主导 consequent
            rules: dict[str, str] = {}
            for av, bc in groups.items():
                support = sum(bc.values())
                if support < min_support:
                    continue
                dom_b, dom_n = max(bc.items(), key=lambda kv: kv[1])
                conf = dom_n / support
                if conf < min_confidence:
                    continue
                base = global_share[b].get(dom_b, 1e-9)
                lift = conf / base if base > 0 else 0.0
                if lift < min_lift:
                    continue
                rules[av] = dom_b
            if not rules:
                continue
            for pos in range(n):
                av, bv = df.iloc[pos][a], df.iloc[pos][b]
                if is_blank(av) or is_blank(bv):
                    continue
                dom_b = rules.get(str(av))
                if dom_b is None or str(bv) == dom_b:
                    continue
                cands.append(CandidateError(
                    row_id=int(df.index[pos]), column=b, value=bv,
                    detector="association_rule", error_type="VAD",
                    score=float(min_confidence),
                    evidence=f"关联规则 {a}={av!r} => {b}={dom_b!r} 被违反(当前 {bv!r})",
                    suggested_fix=dom_b,
                    metadata={"antecedent": a},
                ))
    return cands
