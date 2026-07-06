"""
条件函数依赖(CFD)挖掘（二期）。

CFD 是 FD 的泛化：`(X=x) => A->B` 或 `(X=x) => B=常量`。为避免与一期已落地的
全局 FD（stage_1/fd_detect.py）产生冗余，本模块只挖掘"全局 FD 表达不了、
但在某个低基数条件子集内成立"的依赖，并设**去冗余闸门**：

    仅当条件版一致率显著优于同列对的全局 FD（Δ >= min_gain）才保留，否则丢弃。

两类 CFD：
    1. variable-CFD:  (X=x) => A->B     —— 条件子集内 A 决定 B。
    2. constant-CFD:  (X=x) => B=const  —— 条件子集内 B 恒为某常量（FD 无法表达）。

每条 CFD 经统一分档验证（rule_validation）赋 severity：high 进 mask / medium 仅弱证据 / drop 丢弃。
违反输出 error_type=VAD。仅统计 + 灰区 LLM，空值不参与。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from stage_1.fd_detect import discover_fd, fd_confidence_on
from stage_1.profiling import is_blank


CFD_AUDIT_PROMPT_GRADED = """你是数据质量专家。下面是从一张表中统计挖掘出的"条件函数依赖"，
即仅在满足某条件时才成立的依赖关系。请判断它在现实语义上的可信程度。

条件依赖: {desc}
条件: 当 {cond_col} = '{cond_val}' 时
样例映射:
{samples}

判断标准与三档输出:
- high  : 该条件依赖在现实中确实成立（如"当国家=US 时，ZipCode->State"），违反者应视为错误。
- medium: 可能成立但拿不准（样例不足/弱相关），保留为弱证据但不作确定性结论。
- drop  : 伪依赖（仅因条件子集内某列取值巧合集中而"碰巧"一致）。

只输出 JSON（不要任何解释）:
{{"tier": "high|medium|drop", "reason": "简短理由"}}"""


def _global_confidence(df: pd.DataFrame, a_col: str, b_col: str) -> float:
    """全表上 A->B 的一致率（用于去冗余对比；无有效行返回 0）。"""
    conf = fd_confidence_on(df, a_col, b_col, np.arange(len(df)))
    return conf if conf is not None else 0.0


def _low_card_columns(df: pd.DataFrame, max_card: int) -> list[str]:
    """低基数列（2 <= 非空唯一值 <= max_card），作为候选条件列。"""
    out = []
    for col in df.columns:
        non_blank = df[col][~df[col].map(is_blank)].astype(str)
        card = non_blank.nunique()
        if 2 <= card <= max_card:
            out.append(col)
    return out


def _top_values(series: pd.Series, k: int, min_support: int) -> list[str]:
    """取高频取值（去空），仅保留支持度达标者。"""
    counts = Counter(str(v) for v in series if not is_blank(v))
    return [val for val, cnt in counts.most_common(k) if cnt >= min_support]


def detect_cfd_graded(
    df: pd.DataFrame,
    *,
    cfg,
    thresholds,
    llm=None,
    cache=None,
) -> list[dict]:
    """
    分档版 CFD 检测：挖掘条件依赖 -> 去冗余闸门 -> 统一分档验证 -> 生成 VAD 错误。

    cfg: CFDConfig；thresholds: rule_validation.GradeThresholds。
    返回错误 dict 列表，每条带 `severity`；drop 的 CFD 不产生错误。
    """
    from stage_1.rule_validation import grade_rule

    n_rows = len(df)
    if n_rows == 0:
        return []

    cond_cols = _low_card_columns(df, cfg.max_condition_cardinality)
    errors: list[dict] = []
    n_high = n_med = n_drop = 0

    for cond_col in cond_cols:
        for cond_val in _top_values(df[cond_col], cfg.top_condition_values, cfg.min_subset_support):
            submask = df[cond_col].astype(str) == cond_val
            subset = df[submask]
            if len(subset) < cfg.min_subset_support:
                continue
            sub_index = subset.index.to_numpy()

            # ---- 1) variable-CFD: (cond) => A->B，仅保留显著优于全局 FD 的 ---- #
            for a_col in df.columns:
                if a_col == cond_col:
                    continue
                for b_col in df.columns:
                    if b_col in (cond_col, a_col):
                        continue
                    fd = discover_fd(
                        subset, a_col, b_col,
                        min_confidence=cfg.min_confidence,
                        min_group_support=5,
                        min_group_confidence=cfg.min_confidence,
                    )
                    if fd is None:
                        continue
                    # 去冗余闸门：条件版须显著优于同列对的全局 FD
                    global_conf = _global_confidence(df, a_col, b_col)
                    if fd.confidence - global_conf < cfg.min_gain:
                        continue

                    support = sum(size for _, _, size in fd.mapping.values())
                    samples = "\n".join(
                        f"  {a!r} => {dom!r} ({share:.0%}, n={size})"
                        for a, (dom, share, size) in list(fd.mapping.items())[:12]
                    )
                    desc = f"cfd:({cond_col}={cond_val})->{a_col}->{b_col}"
                    audit_profile = {"_task": "cfd_audit", "desc": desc, "samples": samples}
                    prompt = CFD_AUDIT_PROMPT_GRADED.format(
                        desc=f"{a_col} -> {b_col}", cond_col=cond_col, cond_val=cond_val,
                        samples=samples,
                    )
                    severity, _stab, reason = grade_rule(
                        support=support, confidence=fd.confidence, n_rows=len(subset),
                        th=thresholds,
                        recompute_confidence=(
                            lambda idx, a=a_col, b=b_col, s=subset: fd_confidence_on(s, a, b, idx)
                        ),
                        llm=llm, cache=cache,
                        audit_profile=audit_profile, audit_prompt=prompt,
                        label=desc,
                    )
                    if severity == "drop":
                        n_drop += 1
                        continue
                    n_high += severity == "high"
                    n_med += severity == "medium"
                    for real_row_id, a_val, b_val in zip(
                        sub_index, subset[a_col].to_numpy(), subset[b_col].to_numpy()
                    ):
                        if is_blank(a_val) or is_blank(b_val):
                            continue
                        entry = fd.mapping.get(str(a_val))
                        if entry is None or str(b_val) == entry[0]:
                            continue
                        dom_b, dom_share, group_size = entry
                        errors.append({
                            "row_id": real_row_id, "column": b_col, "value": b_val,
                            "error_type": "VAD",
                            "violated_rule": f"cfd:({cond_col}={cond_val})->{a_col}->{b_col}",
                            "reason": (
                                f"违反条件依赖 当{cond_col}='{cond_val}'时 {a_col}->{b_col}: "
                                f"{a_col}='{a_val}' 多数 {b_col}='{dom_b}'({dom_share:.0%}), 此处='{b_val}'"
                            ),
                            "suggested_fix": dom_b,
                            "confidence": round(dom_share, 3),
                            "severity": severity,
                        })

            # ---- 2) constant-CFD: (cond) => B=const（FD 无法表达） ---- #
            if cfg.enable_constant_cfd:
                for b_col in df.columns:
                    if b_col == cond_col:
                        continue
                    errors_c, sev_c = _detect_constant_cfd(
                        df, subset, sub_index, cond_col, cond_val, b_col,
                        cfg=cfg, thresholds=thresholds, llm=llm, cache=cache,
                    )
                    if sev_c == "high":
                        n_high += 1
                    elif sev_c == "medium":
                        n_med += 1
                    elif sev_c == "drop":
                        n_drop += 1
                    errors.extend(errors_c)

    print(f"条件依赖检测: high {n_high} / medium {n_med} / drop {n_drop}，共 {len(errors)} 个候选错误 (CFD/VAD)")
    return errors


def _detect_constant_cfd(
    df: pd.DataFrame,
    subset: pd.DataFrame,
    sub_index: np.ndarray,
    cond_col: str,
    cond_val: str,
    b_col: str,
    *,
    cfg,
    thresholds,
    llm,
    cache,
) -> tuple[list[dict], Optional[str]]:
    """(X=x)=>B=const：条件子集内 B 恒为常量，且该常量并非全局常量（否则退化为无条件）。"""
    from stage_1.rule_validation import grade_rule

    vals = [str(v) for v in subset[b_col] if not is_blank(v)]
    if len(vals) < cfg.min_subset_support:
        return [], None
    counter = Counter(vals)
    const_val, cnt = counter.most_common(1)[0]
    share = cnt / len(vals)
    if share < cfg.min_confidence:
        return [], None
    # 去冗余：若全局该常量也占主导，则不是"条件化"的，交给列级规则，不重复
    global_vals = [str(v) for v in df[b_col] if not is_blank(v)]
    global_share = (Counter(global_vals).get(const_val, 0) / len(global_vals)) if global_vals else 0.0
    if global_share >= cfg.min_confidence - cfg.min_gain:
        return [], None

    def _recompute(idx: np.ndarray) -> Optional[float]:
        arr = subset[b_col].to_numpy()
        sel = [str(arr[p]) for p in idx if not is_blank(arr[p])]
        if not sel:
            return None
        return sel.count(const_val) / len(sel)

    desc = f"cfd:({cond_col}={cond_val})->{b_col}={const_val}"
    audit_profile = {"_task": "cfd_const_audit", "desc": desc}
    prompt = CFD_AUDIT_PROMPT_GRADED.format(
        desc=f"{b_col} 恒为 '{const_val}'", cond_col=cond_col, cond_val=cond_val,
        samples=f"  {b_col} => '{const_val}' ({share:.0%}, n={cnt}); 全局占比 {global_share:.0%}",
    )
    severity, _stab, reason = grade_rule(
        support=cnt, confidence=share, n_rows=len(subset), th=thresholds,
        recompute_confidence=_recompute, llm=llm, cache=cache,
        audit_profile=audit_profile, audit_prompt=prompt, label=desc,
    )
    if severity == "drop":
        return [], "drop"
    errors = []
    for real_row_id, b_val in zip(sub_index, subset[b_col].to_numpy()):
        if is_blank(b_val) or str(b_val) == const_val:
            continue
        errors.append({
            "row_id": real_row_id, "column": b_col, "value": b_val,
            "error_type": "VAD",
            "violated_rule": f"cfd:({cond_col}={cond_val})->{b_col}=const",
            "reason": (
                f"违反条件常量约束 当{cond_col}='{cond_val}'时 {b_col} 多数为 "
                f"'{const_val}'({share:.0%}), 此处='{b_val}'"
            ),
            "suggested_fix": const_val,
            "confidence": round(share, 3),
            "severity": severity,
        })
    return errors, severity
