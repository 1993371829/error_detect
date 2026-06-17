"""
不一致表示 / 标准化检测（借鉴 Cocoon 的 String Outliers + 列标准化）。

针对"同一概念存在多种写法"的系统性格式/单位不一致错误，例如:
    ounces: '12.0 oz.' / '12.0 ounce' / '12.0 OZ.'  应统一为 '12.0 oz'
    abv:    '0.09%'                                    应统一为 '0.09'

这类错误的关键特征是 **错误形态往往是多数派**，因此无法被"少数派异常"假设的
规则层（受 max_violation_rate 过滤）或分布层捕获。本检测器让 LLM 审阅列内高频
distinct 值，归纳出"规范化规格"（value_map / normalize_regex），再逐格比对
canonical 与 raw，凡不一致即标记为错误（FI），并给出规范写法作为 suggested_fix。

本检测器 **刻意不经过 max_violation_rate 过滤**，但用 max_flag_rate 闸门兜底，
防止 LLM 误把罕见写法当成规范形态而导致整列爆量误报。
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from stage_1.llm_rules import complete_json
from stage_1.profiling import is_blank

STANDARDIZE_PROMPT = """你是数据质量专家。下面是某一列的高频取值（值 => 出现次数）。
请判断这一列是否存在"同一概念的不一致表示"，即同一含义被写成了多种形态，例如:
- 单位/缩写不一致: "12.0 oz." / "12.0 ounce" / "12.0 OZ." 应统一为 "12.0 oz"
- 符号后缀不一致: "0.09%" 应去掉百分号写成 "0.09"
- 大小写/空格不一致: "New York" / "new york" / "NEW YORK"

## 列名: {column}
## 高频取值（值 => 次数）:
{samples}

## 任务
若存在上述不一致，请给出把"非规范写法"转成"规范写法"的标准化规格。规范写法应选择
该概念**语义正确/最规范**的形态（通常是更简洁、更标准的单位或去掉多余符号），不一定是最高频的。

可用两种机制（可同时给出，先 value_map 精确命中，再按顺序套 normalize_regex）:
- value_map: 可枚举的整值映射，形如 {{"12.0 oz.": "12.0 oz", "12.0 ounce": "12.0 oz"}}
  仅写需要被修正的"非规范值"，已规范的值不要写。
- normalize_regex: 顺序应用的正则替换（用于长尾），形如
  [{{"pattern": "%\\\\s*$", "replacement": ""}}]  // 去掉结尾百分号

## 重要约束
1. 只在确有"同一概念多种写法"的系统性不一致时才标准化；自由文本、人名、地址、
   唯一标识等不要标准化，此时返回 needs_standardization=false。
2. 规范化后不应改变语义，只统一表示形态。不要把不同概念合并。
3. normalize_regex 的 pattern 要精确，避免误伤合法值。

## 输出 JSON (只输出 JSON, 不要任何解释)
{{
  "needs_standardization": true,
  "value_map": {{"12.0 oz.": "12.0 oz"}},
  "normalize_regex": [{{"pattern": "%\\\\s*$", "replacement": ""}}],
  "reason": "ounces 列单位写法不一致，统一为 oz"
}}"""


def top_value_samples(series: pd.Series, sample_n: int) -> list[tuple[str, int]]:
    """取列内非空值按频次降序的 top-N (值, 次数)。"""
    non_blank = series[~series.map(is_blank)].astype(str)
    if non_blank.empty:
        return []
    return list(Counter(non_blank).most_common(sample_n))


def build_standardize_profile(column: str, samples: list[tuple[str, int]]) -> dict:
    """构造用于缓存键与 prompt 的轻量画像（命名空间标记防止与规则缓存冲突）。"""
    return {
        "_task": "standardize",
        "column": str(column),
        "values": [[v, int(c)] for v, c in samples],
    }


def extract_canonicalization_spec(llm, column: str, samples: list[tuple[str, int]]) -> dict:
    """
    调用 LLM 归纳该列的标准化规格。

    解析失败或调用异常时返回 needs_standardization=false（不引入误报）。
    """
    if not samples:
        return {"needs_standardization": False}
    sample_lines = "\n".join(f"  {v!r} => {c}" for v, c in samples)
    prompt = STANDARDIZE_PROMPT.format(column=column, samples=sample_lines)
    # 复用规则归纳同款健壮解析（max_tokens 截断保护 + 宽松解析 + 重试）
    spec = complete_json(llm, prompt, label=f"列 {column} 标准化")
    if isinstance(spec, dict):
        return spec
    print(f"[std-warn] 列 {column} 标准化规格解析失败，跳过")
    return {"needs_standardization": False}


def _compile_normalizers(spec: dict):
    """把 spec 编译为 (value_map, [(compiled_pattern, replacement)])。非法正则会被丢弃。"""
    value_map = spec.get("value_map") or {}
    if not isinstance(value_map, dict):
        value_map = {}
    value_map = {str(k): str(v) for k, v in value_map.items()}

    regexes = []
    for item in spec.get("normalize_regex") or []:
        if not isinstance(item, dict):
            continue
        pat = item.get("pattern")
        rep = item.get("replacement", "")
        if not pat:
            continue
        try:
            regexes.append((re.compile(pat), str(rep)))
        except re.error:
            print(f"[std-warn] 非法标准化正则，跳过: {pat}")
    return value_map, regexes


def _canonicalize(value: str, value_map: dict, regexes: list) -> str:
    """先 value_map 精确命中，再依次套正则替换，返回规范形态。"""
    if value in value_map:
        return value_map[value]
    out = value
    for pattern, rep in regexes:
        out = pattern.sub(rep, out)
    return out


def detect_inconsistencies(
    series: pd.Series,
    column: str,
    spec: dict,
    *,
    max_flag_rate: float = 0.95,
) -> list[dict]:
    """
    依据标准化规格逐格检测不一致表示。

    Args:
        series: 待检测列
        column: 列名
        spec: extract_canonicalization_spec 返回的规格
        max_flag_rate: 命中率上限闸门；若 spec 命中比例超过它，判定为 LLM 误选规范形态，
            整列跳过（防爆量误报）。

    Returns:
        错误记录 dict 列表（error_type=FI），canonical != raw 的非空格。
    """
    if not spec or not spec.get("needs_standardization"):
        return []
    value_map, regexes = _compile_normalizers(spec)
    if not value_map and not regexes:
        return []

    non_blank = series[~series.map(is_blank)]
    total = int(len(non_blank))
    if total == 0:
        return []

    reason = str(spec.get("reason", "") or "")
    candidates: list[dict] = []
    for row_id, value in series.items():
        if is_blank(value):
            continue
        raw = str(value)
        canonical = _canonicalize(raw, value_map, regexes)
        if canonical == raw:
            continue
        candidates.append({
            "row_id": row_id,
            "column": column,
            "value": raw,
            "error_type": "FI",
            "violated_rule": "inconsistent_representation",
            "reason": (
                f"不一致表示: '{raw}' 的规范写法应为 '{canonical}'"
                + (f"（{reason}）" if reason else "")
            ),
            "suggested_fix": canonical,
            "confidence": 0.85,
        })

    # 安全闸门：命中率过高通常意味着 LLM 把罕见写法当成了规范形态，整列跳过。
    if candidates and (len(candidates) / total) > max_flag_rate:
        print(
            f"[std-skip] 列 {column} 标准化命中率 {len(candidates)/total:.0%} "
            f"超过 max_flag_rate({max_flag_rate:.0%})，疑似规范形态选错，整列跳过"
        )
        return []
    return candidates
