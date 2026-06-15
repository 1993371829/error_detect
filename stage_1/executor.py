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
from stage_1.fd_detect import detect_vad
from stage_1.llm_rules import LLMClient, extract_rules_for_column
from stage_1.profiling import is_blank, profile_column, save_profiles
from stage_1.rule_cache import RuleCache
from stage_1.rule_compiler import RuleCompiler
from stage_1.rule_guard import should_drop_rule
from stage_1.typo_detect import detect_typos

# 错误记录统一字段顺序（规则类不填 suggested_fix/confidence）
ERROR_COLUMNS = [
    "row_id", "column", "value", "error_type",
    "violated_rule", "reason", "suggested_fix", "confidence",
]


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
        compiled = [compiler.compile(r) for r in exec_rules]
        kept = filter_bad_rules(df, col, compiled, exec_rules, max_violation_rate)

        rule_report.append(
            build_report_entry(
                col,
                rule_spec.get("semantic_type"),
                original_rules,
                kept,
            )
        )

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

    # 跨列/语义检测：Typo(T) 与 函数依赖违反(VAD)
    _append_typo_errors(df, config, all_errors, flagged_cells)
    _append_vad_errors(df, config, all_errors, flagged_cells, rule_report)

    errors_df = pd.DataFrame(all_errors)
    if not errors_df.empty:
        errors_df = errors_df.reindex(columns=ERROR_COLUMNS)
    return errors_df, rule_report


def _append_typo_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
) -> None:
    """运行 Typo 检测并将未被规则层覆盖的单元格并入 all_errors。"""
    tc = config.execution.typo
    if not tc.enabled:
        return
    added = 0
    for col in df.columns:
        typo_errors = detect_typos(
            df[col], str(col),
            max_unique=tc.max_unique,
            min_anchor_count=tc.min_anchor_count,
            rare_max_count=tc.rare_max_count,
            anchor_ratio_min=tc.anchor_ratio_min,
            max_abs_distance=tc.max_abs_distance,
            max_norm_distance=tc.max_norm_distance,
            min_anchor_len=tc.min_anchor_len,
            skip_numeric=tc.skip_numeric,
            numeric_min_ratio=tc.numeric_min_ratio,
        )
        for err in typo_errors:
            cell = (err["row_id"], err["column"])
            if cell in flagged_cells:
                continue
            all_errors.append(err)
            flagged_cells.add(cell)
            added += 1
    print(f"Typo 检测新增 {added} 个候选错误 (T)")


def _append_vad_errors(
    df: pd.DataFrame,
    config: Stage1Config,
    all_errors: list,
    flagged_cells: set,
    rule_report: list,
) -> None:
    """运行 FD 挖掘并将违反依赖的单元格并入 all_errors。"""
    fc = config.execution.fd
    if not fc.enabled:
        return
    vad_errors, fds = detect_vad(
        df,
        min_confidence=fc.min_confidence,
        min_group_support=fc.min_group_support,
        min_group_confidence=fc.min_group_confidence,
        pure_threshold=fc.pure_threshold,
        min_pure_group_ratio=fc.min_pure_group_ratio,
        min_distinct_dependents=fc.min_distinct_dependents,
        max_determinant_unique_ratio=fc.max_determinant_unique_ratio,
        min_dependent_unique=fc.min_dependent_unique,
    )
    added = 0
    for err in vad_errors:
        cell = (err["row_id"], err["column"])
        if cell in flagged_cells:
            continue
        all_errors.append(err)
        flagged_cells.add(cell)
        added += 1
    if fds:
        rule_report.append({
            "discovered_fds": [
                {
                    "determinant": fd.determinant,
                    "dependent": fd.dependent,
                    "confidence": round(fd.confidence, 3),
                    "groups": len(fd.mapping),
                }
                for fd in fds
            ]
        })
        print(f"发现 {len(fds)} 条近似函数依赖, VAD 检测新增 {added} 个错误")
    else:
        print("未发现满足阈值的函数依赖, 跳过 VAD")


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
