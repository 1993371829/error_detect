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
# severity: 双轨可信度分级——high 进 clean_mask（净化训练分布），medium 仅作弱证据不进 mask。
ERROR_COLUMNS = [
    "row_id", "column", "value", "error_type",
    "violated_rule", "reason", "suggested_fix", "confidence", "severity",
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
    cache = RuleCache(config.paths.rule_cache, model=config.llm.model or "")
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

    # 全局确定性检测器链（Typo/温和离群交 Stage 2 多检测器层；FD/CFD/DC 见下方规则族）。
    # 处理顺序：字符级(DMV) -> 列级不一致表示(FI) -> 确定性跨列/范围 -> 极端统计兜底。
    _append_dmv_errors(df, config, all_errors, flagged_cells)
    _append_standardization_errors(df, config, all_errors, flagged_cells, standardize_specs)
    _append_leakage_errors(df, config, all_errors, flagged_cells)
    _append_dup_errors(df, config, all_errors, flagged_cells)
    _append_range_errors(df, config, all_errors, flagged_cells, semantic_types)
    _append_xcol_errors(df, config, all_errors, flagged_cells, llm, cache)
    _append_statistical_extreme_errors(df, config, all_errors, flagged_cells, semantic_types)
    # 规则族（FD/CFD/DC）先收集到独立 sink，不互相去重，供冲突消解按优先级择一。
    # 它们仍跳过已被确定性检测器标记的单元格（flagged_cells），但不写回该集合。
    rule_family_errors: list = []
    # FD 前移：近似函数依赖违反（VAD），经统一分档验证输出双轨 severity。
    _append_fd_errors(df, config, rule_family_errors, flagged_cells, llm, cache)
    # CFD：仅挖全局 FD 漏掉的条件化依赖（对 FD 去冗余），经分档验证输出双轨 VAD。
    _append_cfd_errors(df, config, rule_family_errors, flagged_cells, llm, cache)
    # DC：仅非等值谓词（算术/排序/比较/时序），FD/CFD 结构上表达不了的约束。
    _append_dc_errors(df, config, rule_family_errors, flagged_cells, llm)
    # 规则冲突消解：同单元格多族结论按优先级择一，无法判定者降 medium（不进 mask）。
    from stage_1.rule_conflict import resolve_rule_conflicts
    for err in resolve_rule_conflicts(rule_family_errors):
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        flagged_cells.add(cell)

    errors_df = pd.DataFrame(all_errors)
    if not errors_df.empty:
        # 未显式标注 severity 的记录（既有高精度确定性检测器）默认 high。
        if "severity" not in errors_df.columns:
            errors_df["severity"] = "high"
        else:
            errors_df["severity"] = errors_df["severity"].fillna("high")
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
    # 用此前检测器已标记的单元格构造"部分干净掩码"估参：中位数/MAD/IQR 不被
    # 已知脏值（缺失哨兵替身、极端占位数字等）拉偏；未知脏值仍在，但估参
    # 用的是鲁棒统计量，残余污染影响有限。
    partial_mask = pd.DataFrame(True, index=df.index, columns=df.columns)
    for row_id, col in flagged_cells:
        if row_id in partial_mask.index and col in partial_mask.columns:
            partial_mask.at[row_id, col] = False
    cands = detect_statistical(
        df, partial_mask, ctx,
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


def _grade_thresholds(config: Stage1Config):
    """从配置构造规则分档阈值（FD/CFD/DC 共用）。"""
    from stage_1.rule_validation import GradeThresholds
    v = config.execution.validation
    return GradeThresholds(
        high_min_support=v.high_min_support,
        high_min_confidence=v.high_min_confidence,
        high_min_stability=v.high_min_stability,
        drop_max_support=v.drop_max_support,
        drop_max_confidence=v.drop_max_confidence,
        bootstrap_rounds=v.bootstrap_rounds,
        bootstrap_ratio=v.bootstrap_ratio,
        stability_conf_floor=v.stability_conf_floor,
    )


def _append_fd_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    llm=None,
    cache=None,
) -> None:
    """
    近似函数依赖(FD)违反检测，前移至 Stage 1，经统一分档验证输出双轨 severity：

        - 统计强（高支持 + 高一致率 + 高稳定）-> high（进 clean_mask）。
        - 灰区 -> LLM 三档审核（high/medium/drop）。
        - 统计弱 -> drop（不产生错误）。

    沿用 flagged_cells 去重，规则层已标记的单元格不重复标记。
    """
    fc = config.execution.fd
    if not fc.enabled:
        return
    from stage_1.fd_detect import detect_vad_graded

    vad_errors = detect_vad_graded(
        df,
        thresholds=_grade_thresholds(config),
        min_confidence=fc.min_confidence,
        min_group_support=fc.min_group_support,
        min_group_confidence=fc.min_group_confidence,
        pure_threshold=fc.pure_threshold,
        min_pure_group_ratio=fc.min_pure_group_ratio,
        min_distinct_dependents=fc.min_distinct_dependents,
        max_determinant_unique_ratio=fc.max_determinant_unique_ratio,
        min_dependent_unique=fc.min_dependent_unique,
        llm=llm,
        cache=cache,
    )
    added = 0
    for err in vad_errors:
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        added += 1
    print(f"函数依赖检测新增 {added} 个候选错误 (VAD, 分档验证)")


def _append_cfd_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    llm=None,
    cache=None,
) -> None:
    """
    条件函数依赖(CFD)检测：只挖全局 FD 漏掉的条件化依赖（去冗余闸门内建于 cfd_detect），
    经统一分档验证输出双轨 severity。沿用 flagged_cells 去重。
    """
    cc = config.execution.cfd
    if not cc.enabled:
        return
    from stage_1.cfd_detect import detect_cfd_graded

    cfd_errors = detect_cfd_graded(
        df, cfg=cc, thresholds=_grade_thresholds(config), llm=llm, cache=cache,
    )
    added = 0
    for err in cfd_errors:
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        added += 1
    print(f"条件依赖检测新增 {added} 个候选错误 (CFD/VAD)")


def _append_dc_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    llm=None,
) -> None:
    """
    否定约束(DC)检测：仅非等值谓词（算术/排序/比较/时序），FD/CFD 表达不了的结构约束。
    沿用 flagged_cells 去重。
    """
    dc = config.execution.dc
    if not dc.enabled:
        return
    from stage_1.dc_detect import detect_dc

    dc_errors = detect_dc(df, cfg=dc, thresholds=_grade_thresholds(config), llm=llm)
    added = 0
    for err in dc_errors:
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        added += 1
    print(f"否定约束检测新增 {added} 个候选错误 (DC: FI/VAD)")


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

    双轨策略：仅 severity==high（高可信确定性错误）的单元格进 mask（置 False），
    medium 级（如纯统计近似 FD）不进 mask，避免用从脏数据挖出、易过报的规则
    污染 Stage 2 的训练分布。缺失 severity 视为 high（向后兼容既有产物）。

    供 Stage 2 在干净子集上训练分布模型使用。
    """
    mask = pd.DataFrame(True, index=df.index, columns=df.columns)
    if errors_df is not None and not errors_df.empty:
        if "severity" in errors_df.columns:
            sev = errors_df["severity"].fillna("high").astype(str).str.strip().str.lower()
            high_errors = errors_df[sev != "medium"]
        else:
            high_errors = errors_df
        for row_id, col in zip(high_errors["row_id"], high_errors["column"]):
            if row_id in mask.index and col in mask.columns:
                mask.at[row_id, col] = False
    return mask
