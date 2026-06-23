"""
Step 4: 规则执行 + 错误标记。

编排完整 Stage 1 流水线:
    1. 逐列生成画像
    2. 从缓存或 LLM 获取规则
    3. 编译规则并做误报过滤
    4. 逐单元格执行，输出错误记录

输出:
    - errors DataFrame: row_id, column, value, error_type, violated_rule, reason
    - rule_report: 每列保留的规则及 semantic_type（写入 rules.json）
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import pandas as pd

from stage_1.config import Stage1Config
from stage_1.dmv_detect import detect_dmv
from stage_1.dup_detect import detect_duplicates
from stage_1.leakage_detect import detect_leakage
from stage_1.llm_rules import LLMClient, extract_rules_for_column
from stage_1.profiling import is_blank, profile_column, save_profiles
from stage_1.rule_cache import RuleCache
from stage_1.rule_compiler import RuleCompiler
from stage_1.rule_guard import should_drop_rule
from stage_1.standardize_detect import (
    build_standardize_profile,
    detect_inconsistencies,
    extract_canonicalization_spec,
    top_value_samples,
)
from stage_1.xcol_detect import (
    build_xcol_profile,
    detect_swaps,
    discover_abbrev_pairs,
    validate_abbrev_pair,
)

# 错误记录统一字段顺序（规则类不填 suggested_fix/confidence）
ERROR_COLUMNS = [
    "row_id", "column", "value", "error_type",
    "violated_rule", "reason", "suggested_fix", "confidence",
]

# 可执行类型一致性校验的逻辑类型（其余如 categorical/string 不触发校验）
_CHECKABLE_LOGICAL_TYPES = {
    "bool", "boolean", "int", "integer", "float", "numeric", "number", "date",
}


def _build_type_rule(rule_spec: dict) -> dict | None:
    """
    从 LLM 推断的 logical_type 派生一条"逻辑类型一致性"规则（error_type=FI）。

    仅对 bool/int/float/date 生成规则；其余类型返回 None（不校验）。
    """
    logical = str(rule_spec.get("logical_type", "") or "").strip().lower()
    if logical not in _CHECKABLE_LOGICAL_TYPES:
        return None
    return {
        "type": "logical_type",
        "spec": {"logical_type": logical},
        "error_type": "FI",
        "reason": f"列逻辑类型应为 {logical}，此值不符合该类型",
    }


def scan_missing_value(row_id, column: str, value) -> dict | None:
    """
    独立缺失值扫描：不依赖 LLM 是否生成 not_null。
    所有空串/NaN/empty 等缺失哨兵 -> 记为 MV。
    """
    if is_blank(value):
        return {
            "row_id": row_id,
            "column": column,
            "value": value,
            "error_type": "MV",
            "violated_rule": "missing_value",
            "reason": "缺失值：缺失哨兵(空串/NaN/empty 等)",
        }
    return None


def is_nullable_column(
    col: str,
    profile: dict,
    original_rules: list,
    config: Stage1Config,
) -> bool:
    """判断该列是否跳过 MV 扫描（allow_blank）。"""
    if col in config.execution.nullable_columns:
        return True
    if config.execution.infer_nullable:
        has_not_null = any(r.get("type") == "not_null" for r in original_rules)
        if (
            not has_not_null
            and profile.get("null_rate", 0) >= config.execution.infer_nullable_min_rate
        ):
            return True
    return False


def filter_bad_rules(
    df: pd.DataFrame,
    col: str,
    compiled_rules: list,
    raw_rules: list,
    max_violation_rate: float = 0.3,
) -> list:
    """
    误报控制：丢弃违反率过高的规则。

    若某规则导致超过 max_violation_rate 比例的行报错，
    说明 LLM 可能过度拟合或规则过严，应丢弃。

    (MV 由 executor 的独立缺失值扫描负责，不在此处处理)
    """
    kept = []
    values = df[col].tolist()
    for cf, raw in zip(compiled_rules, raw_rules):
        violations = sum(1 for v in values if cf(v) is not None)
        rate = violations / len(values) if values else 0
        if rate <= max_violation_rate:
            kept.append((cf, raw))
        else:
            print(
                f"[discard] 列 {col} 规则 {raw.get('type')} 违反率 {rate:.0%} 过高,丢弃 "
                f"(reason: {raw.get('reason', '')})"
            )
    return kept


def build_report_entry(
    col: str,
    semantic_type,
    original_rules: list,
    kept: list,
) -> dict:
    """
    构造单列表报告条目。

    rules_kept     : 实际参与执行的规则（已去 not_null、过 guard、过违反率）
    semantic_notes : 语义上记录但不执行的规则（如 not_null）
    """
    executed_raw = [raw for _, raw in kept]
    semantic_notes = [
        {
            **r,
            "enforced": False,
            "note": "MV 由独立缺失值扫描负责，此规则仅作语义记录",
        }
        for r in original_rules
        if r.get("type") == "not_null"
    ]
    return {
        "column": col,
        "semantic_type": semantic_type,
        "rules_kept": executed_raw,
        "semantic_notes": semantic_notes,
    }


def run_rule_layer(
    df: pd.DataFrame,
    config: Stage1Config,
    dry_run: bool = False,
) -> Tuple[pd.DataFrame, list]:
    """
    执行 Stage 1 完整流水线。

    Args:
        df: 输入 DataFrame（建议 dtype=str）
        config: Stage1Config 配置对象
        dry_run: True 时仅打印画像，不调用 LLM、不执行规则

    Returns:
        (errors_df, rule_report)
        - errors_df: 错误单元格记录，dry_run 时为空 DataFrame
        - rule_report: 每列归纳规则列表，供写入 rules.json
    """
    compiler = RuleCompiler()
    cache = RuleCache(config.paths.rule_cache)
    llm: Optional[LLMClient] = None if dry_run else LLMClient(config)

    all_errors = []
    flagged_cells: set = set()  # 已被规则层标记的 (row_id, column)，供后续去重
    rule_report = []
    profiles = []
    standardize_specs: dict = {}  # col -> LLM 标准化规格（供后续不一致检测）
    semantic_types: dict = {}     # col -> semantic_type（供硬范围规则使用）
    max_violation_rate = config.execution.max_violation_rate
    max_samples = config.profiling.max_samples

    for col in df.columns:
        profile = profile_column(df[col], max_samples=max_samples)
        profiles.append(profile)

        if dry_run:
            print(f"\n=== 列: {col} ===")
            print(json.dumps(profile, ensure_ascii=False, indent=2, default=str))
            continue

        # 优先读缓存，miss 时调用 LLM 并写回
        rule_spec = cache.get(profile)
        if rule_spec is None:
            rule_spec = extract_rules_for_column(llm, profile)
            cache.set(profile, rule_spec)

        original_rules = rule_spec.get("rules", [])
        exec_rules = [
            r
            for r in original_rules
            if r.get("type") != "not_null"
            and not should_drop_rule(r, profile, col, config.execution.guard)
        ]
        # 列逻辑类型一致性校验（借鉴 Cocoon）：由 LLM 推断的 logical_type 派生一条类型规则，
        # 与其他规则一同经 max_violation_rate 兜底过滤，避免误判类型造成系统性误报。
        if config.execution.enable_type_check:
            type_rule = _build_type_rule(rule_spec)
            if type_rule is not None:
                exec_rules = exec_rules + [type_rule]
        compiled = [compiler.compile(r) for r in exec_rules]
        kept = filter_bad_rules(df, col, compiled, exec_rules, max_violation_rate)

        semantic_types[col] = rule_spec.get("semantic_type", "")
        rule_report.append(
            build_report_entry(
                col,
                rule_spec.get("semantic_type"),
                original_rules,
                kept,
            )
        )

        # 不一致表示/标准化规格（借鉴 Cocoon String Outliers）：让 LLM 审阅高频取值，
        # 归纳"同一概念多种写法"的标准化规格，复用规则缓存文件（命名空间键避免冲突）。
        # 跳过高基数自由文本/标识列，控制误报。
        sc = config.execution.standardize
        if sc.enabled:
            non_blank_n = profile.get("total_count", 0) - profile.get("null_count", 0)
            distinct_ratio = (profile.get("unique_count", 0) / non_blank_n) if non_blank_n else 1.0
            if distinct_ratio <= sc.max_distinct_ratio:
                samples = top_value_samples(df[col], sc.sample_n)
                std_profile = build_standardize_profile(str(col), samples)
                std_spec = cache.get(std_profile)
                if std_spec is None:
                    std_spec = extract_canonicalization_spec(llm, str(col), samples)
                    cache.set(std_profile, std_spec)
                if std_spec.get("needs_standardization"):
                    standardize_specs[col] = std_spec

        allow_blank = is_nullable_column(col, profile, original_rules, config)

        # 逐单元格执行，每格只报告第一条违反的规则
        for idx, value in df[col].items():
            if not allow_blank:
                mv = scan_missing_value(idx, col, value)
                if mv is not None:
                    all_errors.append(mv)
                    flagged_cells.add((idx, col))
                    continue

            for cf, raw in kept:
                result = cf(value)
                if result:
                    error_type, rule = result
                    all_errors.append({
                        "row_id": idx,
                        "column": col,
                        "value": value,
                        "error_type": error_type,
                        "violated_rule": rule.get("type"),
                        "reason": rule.get("reason", ""),
                    })
                    flagged_cells.add((idx, col))
                    break

    if dry_run:
        path = save_profiles(profiles, config.paths.profiles_output)
        print(f"\n列画像已写入: {path}")
        return pd.DataFrame(all_errors), rule_report

    # Stage 1 仅保留高精度规则/确定性检测；Typo/主导格式/FD 已迁移至 Stage 2 多检测器层。
    # 处理顺序：字符级(DMV) -> 列级不一致表示(FI) -> 确定性跨列/范围 -> 极端统计兜底。
    _append_dmv_errors(df, config, all_errors, flagged_cells)
    _append_standardization_errors(df, config, all_errors, flagged_cells, standardize_specs)
    _append_leakage_errors(df, config, all_errors, flagged_cells)
    _append_dup_errors(df, config, all_errors, flagged_cells)
    _append_range_errors(df, config, all_errors, flagged_cells, semantic_types)
    _append_iforest_errors(df, config, all_errors, flagged_cells)
    _append_xcol_errors(df, config, all_errors, flagged_cells, llm, cache)
    _append_arith_errors(df, config, all_errors, flagged_cells, llm)
    _append_statistical_extreme_errors(df, config, all_errors, flagged_cells, semantic_types)

    errors_df = pd.DataFrame(all_errors)
    if not errors_df.empty:
        errors_df = errors_df.reindex(columns=ERROR_COLUMNS)
    return errors_df, rule_report


def _append_dmv_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
) -> None:
    """运行伪缺失值(DMV)检测并将未被覆盖的单元格并入 all_errors。"""
    dc = config.execution.dmv
    if not dc.enabled:
        return
    added = 0
    for col in df.columns:
        dmv_errors = detect_dmv(
            df[col], str(col),
            extra_tokens=dc.extra_tokens or None,
            detect_numeric_placeholder=dc.detect_numeric_placeholder,
            numeric_placeholders=dc.numeric_placeholders or None,
        )
        for err in dmv_errors:
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"伪缺失值检测新增 {added} 个候选错误 (DMV)")


def _append_standardization_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    standardize_specs: dict,
) -> None:
    """
    依据逐列 LLM 标准化规格检测"不一致表示"，并将未被覆盖的单元格并入 all_errors。

    刻意不经过 max_violation_rate（这类错误本就是多数派），由检测器内部的
    max_flag_rate 闸门兜底防爆量误报。
    """
    sc = config.execution.standardize
    if not sc.enabled or not standardize_specs:
        return
    added = 0
    for col, spec in standardize_specs.items():
        std_errors = detect_inconsistencies(
            df[col], str(col), spec, max_flag_rate=sc.max_flag_rate,
        )
        for err in std_errors:
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"标准化检测新增 {added} 个候选错误 (FI/不一致表示)")


def _append_leakage_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
) -> None:
    """运行引用/元数据泄漏检测，将未被覆盖的单元格并入 all_errors（交 Stage 3 核验）。"""
    lc = config.execution.leakage
    if not lc.enabled:
        return
    added = 0
    for col in df.columns:
        leak_errors = detect_leakage(
            df[col], str(col),
            skip_numeric=lc.skip_numeric,
            numeric_min_ratio=lc.numeric_min_ratio,
            min_len_ratio=lc.min_len_ratio,
        )
        for err in leak_errors:
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"元数据泄漏检测新增 {added} 个候选错误 (FI/metadata_leakage)")


def _append_dup_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
) -> None:
    """运行重复值检测（整值由同一 token 重复拼接），并入 all_errors（交 Stage 3 核验）。"""
    dc = config.execution.dup
    if not dc.enabled:
        return
    added = 0
    for col in df.columns:
        for err in detect_duplicates(df[col], str(col), sep=dc.sep):
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"重复值检测新增 {added} 个候选错误 (FI/duplicate_value)")


def _append_statistical_extreme_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    semantic_types: dict,
) -> None:
    """
    极端统计兜底（高精度）：数值/日期列用 Robust-Z(>6) + IQR(k=4.5) 双判据标记极端离群值。

    阈值远高于 Stage 2 统计检测器（z>3），只捞确凿的极端值，作为 Stage 1 的确定性补充；
    温和离群交由 Stage 2 多检测器 + 证据融合处理。
    """
    sc = config.execution.statistic_extreme
    if not sc.enabled:
        return
    from stage_2.coltypes import infer_column_kinds
    from stage_2.detectors.base import DetectorContext
    from stage_2.detectors.statistical import detect_statistical

    kinds = infer_column_kinds(df)
    ctx = DetectorContext(kinds=kinds, semantic_types=semantic_types or {})
    cands = detect_statistical(
        df, None, ctx,
        robust_z=sc.robust_z, iqr_k=sc.iqr_k, detect_dates=sc.detect_dates,
    )
    added = 0
    for cand in cands:
        cell = (cand.row_id, cand.column)
        if cell in flagged_cells:
            continue
        all_errors.append({
            "row_id": cand.row_id,
            "column": cand.column,
            "value": cand.value,
            "error_type": "FI",
            "violated_rule": "extreme_outlier",
            "reason": cand.evidence,
        })
        flagged_cells.add(cell)
        added += 1
    print(f"极端统计检测新增 {added} 个候选错误 (FI/extreme_outlier)")


def _append_range_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    semantic_types: dict,
) -> None:
    """基于 semantic_type 的硬范围规则（age/percentage/price/lat/lon 等）。"""
    rc = config.execution.range_check
    if not rc.enabled:
        return
    from stage_1.range_detect import detect_range_errors
    added = 0
    for col in df.columns:
        for err in detect_range_errors(
            df[col], str(col), semantic_types.get(col, ""),
            numeric_min_ratio=rc.numeric_min_ratio,
        ):
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"硬范围检测新增 {added} 个候选错误 (FI/range_error)")


def _append_iforest_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
) -> None:
    """数值列联合孤立森林极端值检测（默认关闭）。"""
    ic = config.execution.iforest
    if not ic.enabled:
        return
    from stage_1.iforest_detect import detect_iforest_outliers
    added = 0
    for err in detect_iforest_outliers(
        df, top_quantile=ic.top_quantile, min_numeric_cols=ic.min_numeric_cols,
    ):
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        flagged_cells.add(cell)
        added += 1
    print(f"孤立森林检测新增 {added} 个候选错误 (FI/extreme_outlier)")


def _append_arith_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    llm=None,
) -> None:
    """LLM 跨列算术规则（AST 安全求值 + 验证；默认关闭）。"""
    ac = config.execution.arith
    if not ac.enabled or llm is None:
        return
    from stage_1.arith_rules import generate_arithmetic_rules, validate_and_apply
    rules = generate_arithmetic_rules(llm, df)
    if not rules:
        print("跨列算术规则：LLM 未归纳出可用规则，跳过")
        return
    added = 0
    for err in validate_and_apply(
        df, rules, min_support=ac.min_support, max_violation_rate=ac.max_violation_rate,
    ):
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        flagged_cells.add(cell)
        added += 1
    print(f"跨列算术规则新增 {added} 个候选错误 (FI/arithmetic_constraint)")


def _append_xcol_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    llm: Optional[LLMClient] = None,
    cache=None,
) -> None:
    """发现"全名↔缩写"列对（LLM 语义确认）并标记两列对调的行（交 Stage 3 复核）。"""
    xc = config.execution.xcol
    if not xc.enabled or llm is None:
        return
    pairs = discover_abbrev_pairs(
        df,
        min_rows=xc.min_rows,
        min_len_ratio=xc.min_len_ratio,
        min_multi_token_rate=xc.min_multi_token_rate,
        min_abbrev_rate=xc.min_abbrev_rate,
    )
    added = 0
    for full_col, abbrev_col, _rate in pairs:
        samples = [
            (str(df[full_col][i]), str(df[abbrev_col][i]))
            for i in range(len(df))
            if not is_blank(df[full_col][i]) and not is_blank(df[abbrev_col][i])
        ][:30]
        if xc.semantic_check:
            confirmed = None
            if cache is not None:
                profile = build_xcol_profile(full_col, abbrev_col, samples)
                cached = cache.get(profile)
                if cached is not None and isinstance(cached, dict):
                    confirmed = bool(cached.get("is_abbreviation_pair"))
            if confirmed is None:
                confirmed = validate_abbrev_pair(llm, full_col, abbrev_col, samples)
                if cache is not None:
                    cache.set(build_xcol_profile(full_col, abbrev_col, samples),
                              {"is_abbreviation_pair": confirmed})
            if not confirmed:
                print(f"[xcol] 列对 {full_col}↔{abbrev_col} 经 LLM 判定非'全名↔缩写'关系，跳过")
                continue
        for err in detect_swaps(df, full_col, abbrev_col):
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"跨列对调检测新增 {added} 个候选错误 (FI/column_swap)")


def build_clean_mask(df: pd.DataFrame, errors_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成与 df 同形状的布尔掩码: True 表示该单元格未被任何检测器标记(干净)。

    供 Stage 2 在干净子集上训练分布模型使用。
    """
    mask = pd.DataFrame(True, index=df.index, columns=df.columns)
    if errors_df is not None and not errors_df.empty:
        for row_id, col in zip(errors_df["row_id"], errors_df["column"]):
            if row_id in mask.index and col in mask.columns:
                mask.at[row_id, col] = False
    return mask
