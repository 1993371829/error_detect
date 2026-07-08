"""
Stage 3 核心：逐行调用 LLM（带缓存）对可疑单元格做精检，解析每格判定。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from stage_3.cache import ResponseCache
from stage_3.context import RowContext, SuspectCell
from stage_3.prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, build_user_prompt

VALID_TYPES = {"MV", "DMV", "T", "VAD", "FI", "OTHER", "NONE"}
# B3 去重：可安全按 (列,值,前序类型) 复用判定的"上下文无关"错误类型
_CONTEXT_FREE_TYPES = {"MV", "DMV", "T", "FI"}


def _cache_key(user_prompt: str) -> str:
    """响应缓存键：叠加 system 版本号，模板变更时自动失效。"""
    return f"{SYSTEM_PROMPT_VERSION}\n{user_prompt}"


def _context_free_eligible(s: SuspectCell) -> bool:
    """该可疑格是否可按 (列,值) 去重判定（与整行上下文无关）。

    仅限确定性/单值可判类型，且不依赖分布模型/跨行共识证据，避免误伤上下文相关判定。
    """
    if s.auto_confirm:
        return False
    if _is_stage2(s) or s.consensus is not None:
        return False
    if s.verifiability != "verifiable":
        return False
    return str(s.prior_error_type).upper() in _CONTEXT_FREE_TYPES
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
        "fix_source": "prior" if s.suggested_fix else "",
        "fix_confidence": 0.5,
        "llm_reason": "LLM 未返回该格判定，沿用前序结果（兜底）。",
    }


def _is_stage2(s: SuspectCell) -> bool:
    return "stage2" in (s.prior_source or "").lower()


def _auto_confirm_judgment(s: SuspectCell) -> dict:
    """高置信确定性结构错误（如 duplicate_value）直接确认，不经 LLM。"""
    etype = s.prior_error_type if s.prior_error_type in VALID_TYPES else "FI"
    return {
        "row_id": None,
        "column": s.column,
        "value": s.value,
        "prior_error_type": s.prior_error_type,
        "prior_source": s.prior_source,
        "is_error": True,
        "error_type": etype,
        "confidence": 0.95,
        "suggested_fix": s.suggested_fix or None,
        "fix_source": "prior_rule" if s.suggested_fix else "",
        "fix_confidence": 0.95,
        "llm_reason": "[直通] Stage1 确定性结构错误，高置信直接确认（跳过 LLM）。",
    }


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
    fix_source = "llm" if fix else ""
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
            "fix_source": fix_source,
            "fix_confidence": round(conf, 3),
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
            if fix:
                fix_source = "consensus"
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
        "fix_source": fix_source,
        "fix_confidence": round(conf, 3),
        "llm_reason": llm_reason,
    }


def verify_row(
    ctx: RowContext,
    llm,
    cache: Optional[ResponseCache],
    reject_conf_threshold: float = DEFAULT_REJECT_CONF_THRESHOLD,
    protect_stage1_mv: bool = True,
    *,
    enable_thinking: Optional[bool] = None,
    memo: Optional[dict] = None,
    memo_lock: Optional[threading.Lock] = None,
) -> list[dict]:
    """对单行构造 prompt、(缓存或)调用 LLM、解析并返回逐格判定。

    - 高置信确定性结构错误（auto_confirm）直接确认；
    - 若 memo 提供（B3 去重开启），上下文无关格命中 (列,值,类型) 记忆则复用、不进 prompt；
    - 若整行可疑格都无需 LLM（全 auto_confirm 或全命中 memo），跳过 LLM 调用。
    """
    memo_hits: dict[int, dict] = {}   # ctx.suspects 下标 -> 复用的判定
    llm_suspects = []
    for s in ctx.suspects:
        if s.auto_confirm:
            continue
        if memo is not None and _context_free_eligible(s):
            key = (s.column, str(s.value), str(s.prior_error_type).upper())
            cached = None
            if memo_lock is not None:
                with memo_lock:
                    cached = memo.get(key)
            else:
                cached = memo.get(key)
            if cached is not None:
                memo_hits[id(s)] = cached
                continue
        llm_suspects.append(s)

    by_col: dict[str, dict] = {}
    if llm_suspects:
        prompt_ctx = RowContext(
            row_id=ctx.row_id, row_values=ctx.row_values, suspects=llm_suspects,
        )
        user_prompt = build_user_prompt(prompt_ctx)
        key = _cache_key(user_prompt)
        raw = cache.get(key) if cache else None
        if raw is None:
            # 单行 LLM 调用失败（内容审查 400 / 限流 / 网络等）不应拖垮整个并发精检：
            # 捕获后对本行回退前序判定（保住召回），继续处理其余行。
            try:
                raw = llm.complete(user_prompt, system=SYSTEM_PROMPT, enable_thinking=enable_thinking)
            except Exception as exc:  # noqa: BLE001
                print(f"  [stage3][warn] row {ctx.row_id} LLM 调用失败，回退前序判定："
                      f"{type(exc).__name__}: {str(exc)[:120]}")
                raw = None
            if raw is not None and cache:
                cache.set(key, raw)
        by_col = _parse_response(raw) if raw is not None else {}

    results = []
    for s in ctx.suspects:
        if s.auto_confirm:
            norm = _auto_confirm_judgment(s)
        elif id(s) in memo_hits:
            norm = dict(memo_hits[id(s)])  # 复用判定，替换本格标识
            norm["column"], norm["value"] = s.column, s.value
            norm["prior_error_type"], norm["prior_source"] = s.prior_error_type, s.prior_source
        else:
            judg = by_col.get(s.column)
            norm = (
                _normalize(judg, s, reject_conf_threshold, protect_stage1_mv) if judg
                else _fallback_judgment(s)
            )
            # 记忆上下文无关格判定，供其他行的相同 (列,值,类型) 复用
            if memo is not None and _context_free_eligible(s) and judg:
                store = dict(norm)
                store["row_id"] = None
                mkey = (s.column, str(s.value), str(s.prior_error_type).upper())
                if memo_lock is not None:
                    with memo_lock:
                        memo.setdefault(mkey, store)
                else:
                    memo.setdefault(mkey, store)
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
    *,
    max_workers: int = 8,
    enable_thinking: Optional[bool] = None,
    dedup_context_free: bool = False,
) -> list[dict]:
    """遍历所有行上下文，返回扁平的逐格判定列表（顺序与输入一致）。

    max_workers>1 时用线程池并发调用 LLM（墙钟大降）；dedup_context_free 开启时
    对上下文无关格按 (列,值,类型) 复用判定（省调用/ token，默认关闭需消融验证）。
    """
    total = len(contexts)
    memo: Optional[dict] = {} if dedup_context_free else None
    memo_lock = threading.Lock() if dedup_context_free else None

    def _run(ctx: RowContext) -> list[dict]:
        return verify_row(
            ctx, llm, cache, reject_conf_threshold, protect_stage1_mv,
            enable_thinking=enable_thinking, memo=memo, memo_lock=memo_lock,
        )

    row_results: list[Optional[list[dict]]] = [None] * total
    done = 0
    _progress_lock = threading.Lock()

    def _tick() -> None:
        nonlocal done
        with _progress_lock:
            done += 1
            d = done
        if progress_every and (d % progress_every == 0 or d == total):
            print(f"  [stage3] 已处理 {d}/{total} 行")

    if max_workers and max_workers > 1 and total > 1:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, ctx): i for i, ctx in enumerate(contexts)}
            for fut in as_completed(futures):
                row_results[futures[fut]] = fut.result()
                _tick()
    else:
        for i, ctx in enumerate(contexts):
            row_results[i] = _run(ctx)
            _tick()

    all_results: list[dict] = []
    for r in row_results:
        if r:
            all_results.extend(r)
    if cache is not None:
        cache.flush()
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
            if not r.get("fix_source"):
                r["fix_source"] = "propagated"
            filled += 1
    if filled:
        print(f"修复映射复用：为 {filled} 个同列同值单元格补全 suggested_fix")
    return results
