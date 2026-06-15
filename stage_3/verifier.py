"""
Stage 3 核心：逐行调用 LLM（带缓存）对可疑单元格做精检，解析每格判定。
"""

from __future__ import annotations

import json
from typing import Optional

from stage_3.cache import ResponseCache
from stage_3.context import RowContext, SuspectCell
from stage_3.prompt import build_prompt

VALID_TYPES = {"MV", "T", "VAD", "FI", "OTHER", "NONE"}
# 前序类型 -> 解析失败时的兜底最终类型
_FALLBACK_TYPE = {"DIST": "OTHER", "": "OTHER"}


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


def _normalize(judg: dict, s: SuspectCell) -> dict:
    """规整单格判定字段，容错缺省值。"""
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
        "llm_reason": str(judg.get("reason", "") or ""),
    }


def verify_row(ctx: RowContext, llm, cache: Optional[ResponseCache]) -> list[dict]:
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
        norm = _normalize(judg, s) if judg else _fallback_judgment(s)
        norm["row_id"] = ctx.row_id
        results.append(norm)
    return results


def verify_contexts(
    contexts: list[RowContext],
    llm,
    cache: Optional[ResponseCache] = None,
    progress_every: int = 50,
) -> list[dict]:
    """遍历所有行上下文，返回扁平的逐格判定列表。"""
    all_results: list[dict] = []
    total = len(contexts)
    for i, ctx in enumerate(contexts, 1):
        all_results.extend(verify_row(ctx, llm, cache))
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  [stage3] 已处理 {i}/{total} 行")
    return all_results
