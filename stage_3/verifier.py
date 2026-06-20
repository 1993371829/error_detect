"""
Stage 3 核心：逐行调用 LLM（带缓存）对可疑单元格做精检，解析每格判定。
"""

from __future__ import annotations

import json
from typing import Optional

from stage_3.cache import ResponseCache
from stage_3.context import RowContext, SuspectCell
from stage_3.prompt import build_prompt

VALID_TYPES = {"MV", "DMV", "T", "VAD", "FI", "OTHER", "NONE"}
# 前序类型 -> 解析失败时的兜底最终类型
_FALLBACK_TYPE = {"DIST": "OTHER", "": "OTHER"}
# 缺证据保护：默认否决置信度门槛（低于此值且证据冲突时不允许否决）
DEFAULT_REJECT_CONF_THRESHOLD = 0.85


def _parse_response(raw: str) -> dict:
    """解析 LLM JSON，返回 列名 -> 判定 dict 的映射；失败返回空。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    judgments = data.get("judgments", []) if isinstance(data, dict) else []
    out = {}
    for j in judgments:
        if isinstance(j, dict) and "column" in j:
            out[str(j["column"])] = j
    return out


def _fallback_judgment(s: SuspectCell) -> dict:
    """LLM 缺该格判定时的保守兜底：维持为错误，置信度中等。"""
    etype = (
        s.prior_error_type if s.prior_error_type in VALID_TYPES
        else _FALLBACK_TYPE.get(s.prior_error_type, "OTHER")
    )
    return {
        "row_id": None,
        "column": s.column,
        "value": s.value,
        "prior_error_type": s.prior_error_type,
        "prior_source": s.prior_source,
        "is_error": True,
        "error_type": etype,
        "confidence": 0.5,
        "suggested_fix": s.suggested_fix or None,
        "llm_reason": "LLM 未返回该格判定，沿用前序结果（兜底）。",
    }


def _is_stage2(s: SuspectCell) -> bool:
    return "stage2" in (s.prior_source or "").lower()


def _is_stage1(s: SuspectCell) -> bool:
    return "stage1" in (s.prior_source or "").lower()


def _normalize(
    judg: dict,
    s: SuspectCell,
    reject_conf_threshold: float = DEFAULT_REJECT_CONF_THRESHOLD,
    protect_stage1_mv: bool = True,
) -> dict:
    """规整单格判定字段，容错缺省值；对跨行共识冲突的候选施加缺证据保护。"""
    etype = str(judg.get("error_type", "") or "").upper()
    is_error = judg.get("is_error")
    if is_error is None:
        is_error = etype not in ("NONE", "")
    if etype not in VALID_TYPES:
        etype = "OTHER" if is_error else "NONE"
    if etype == "NONE":
        is_error = False
    try:
        conf = float(judg.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    fix = judg.get("suggested_fix")
    fix = None if fix in (None, "", "null") else str(fix)
    llm_reason = str(judg.get("reason", "") or "")

    # 确定性缺失值保护：Stage1 标记的 MV 是确定信号（该格确实为空，且 Stage1 已判该列
    # 非空），不允许 LLM 二次否决。经 rayyan/flights/hospital 多数据集验证：救回真错且
    # 零新增误报（MV 的精度本就接近 1.0，LLM 对其的否决在实测中均为错判）。
    if (
        not is_error
        and protect_stage1_mv
        and str(s.prior_error_type).upper() == "MV"
        and _is_stage1(s)
    ):
        is_error = True
        etype = "MV"
        llm_reason = (
            f"[保护] Stage1 确定性缺失值，不被 LLM 否决（该列已判非空）。"
            f"LLM 原判: {llm_reason or 'NONE'}"
        )
        return {
            "row_id": None,
            "column": s.column,
            "value": s.value,
            "prior_error_type": s.prior_error_type,
            "prior_source": s.prior_source,
            "is_error": True,
            "error_type": etype,
            "confidence": round(conf, 3),
            "suggested_fix": fix,
            "llm_reason": llm_reason,
        }

    # 缺证据保护：来自分布模型(stage2)、且跨行共识证据表明本值与同 key 多数值冲突时，
    # 只有当 LLM 以足够高的把握判其为误报，才允许否决；否则维持为错误（保住召回）。
    if (
        not is_error
        and _is_stage2(s)
        and s.consensus_conflict
        and conf < reject_conf_threshold
    ):
        is_error = True
        etype = "VAD" if s.column != s.consensus.get("key_column") else "OTHER"
        if fix is None:
            fix = s.consensus.get("majority_value") or (s.suggested_fix or None)
        llm_reason = (
            f"[保护] LLM 以低把握({conf:.2f}<{reject_conf_threshold})判 NONE，"
            f"但同 {s.consensus.get('key_column')} 多数值="
            f"{s.consensus.get('majority_value')!r}(占比 {s.consensus.get('majority_share')})"
            f"与本值冲突，维持为错误。原因: {llm_reason}"
        )

    return {
        "row_id": None,  # 由调用方填
        "column": s.column,
        "value": s.value,
        "prior_error_type": s.prior_error_type,
        "prior_source": s.prior_source,
        "is_error": bool(is_error),
        "error_type": etype,
        "confidence": round(conf, 3),
        "suggested_fix": fix,
        "llm_reason": llm_reason,
    }


def verify_row(
    ctx: RowContext,
    llm,
    cache: Optional[ResponseCache],
    reject_conf_threshold: float = DEFAULT_REJECT_CONF_THRESHOLD,
    protect_stage1_mv: bool = True,
) -> list[dict]:
    """对单行构造 prompt、(缓存或)调用 LLM、解析并返回逐格判定。"""
    prompt = build_prompt(ctx)
    raw = cache.get(prompt) if cache else None
    if raw is None:
        raw = llm.complete(prompt)
        if cache:
            cache.set(prompt, raw)

    by_col = _parse_response(raw)
    results = []
    for s in ctx.suspects:
        judg = by_col.get(s.column)
        norm = (
            _normalize(judg, s, reject_conf_threshold, protect_stage1_mv) if judg
            else _fallback_judgment(s)
        )
        norm["row_id"] = ctx.row_id
        results.append(norm)
    return results


def verify_contexts(
    contexts: list[RowContext],
    llm,
    cache: Optional[ResponseCache] = None,
    progress_every: int = 50,
    reject_conf_threshold: float = DEFAULT_REJECT_CONF_THRESHOLD,
    protect_stage1_mv: bool = True,
) -> list[dict]:
    """遍历所有行上下文，返回扁平的逐格判定列表。"""
    all_results: list[dict] = []
    total = len(contexts)
    for i, ctx in enumerate(contexts, 1):
        all_results.extend(
            verify_row(ctx, llm, cache, reject_conf_threshold, protect_stage1_mv)
        )
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  [stage3] 已处理 {i}/{total} 行")
    return all_results


def propagate_fix_mappings(results: list[dict]) -> list[dict]:
    """
    借鉴 Cocoon 的 old->new 映射复用：把同一列同一脏值的修复建议在确认错误的单元格间复用，
    保证一致性并补全缺失的 suggested_fix。

    构建 (column, value) -> 最高置信度的 suggested_fix（仅取 is_error 且 fix 非空者），
    再回填那些被确认为错误但 suggested_fix 为空的同 (column, value) 单元格。
    不改变 is_error / error_type，只补全/统一修复值。
    """
    best_fix: dict[tuple, tuple[str, float]] = {}
    for r in results:
        if not r.get("is_error"):
            continue
        fix = r.get("suggested_fix")
        if fix in (None, "", "null"):
            continue
        key = (str(r.get("column")), str(r.get("value")))
        conf = float(r.get("confidence", 0.0) or 0.0)
        prev = best_fix.get(key)
        if prev is None or conf > prev[1]:
            best_fix[key] = (str(fix), conf)

    if not best_fix:
        return results

    filled = 0
    for r in results:
        if not r.get("is_error"):
            continue
        if r.get("suggested_fix") not in (None, "", "null"):
            continue
        key = (str(r.get("column")), str(r.get("value")))
        mapped = best_fix.get(key)
        if mapped is not None:
            r["suggested_fix"] = mapped[0]
            filled += 1
    if filled:
        print(f"修复映射复用：为 {filled} 个同列同值单元格补全 suggested_fix")
    return results
