"""
规则兜底校验：在编译前丢弃过严的自由文本 regex。

适用范围:
    - 仅对自由文本列（HospitalName/Address 等）启用
    - 结构化列（PhoneNumber/MeasureCode/ZipCode 等）完全跳过 guard

判定方式:
    - 用 edge_samples 中的完整值做 pattern.match(value)
    - edge_samples 含含空格多词值，用于捕获禁止空格的过严 regex
    - 不用 charset_info.special_chars（含脏数据噪声，且单字符 match 不准确）

charset_info 仍保留在 profile 中，仅供 LLM 提示词使用。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from stage_1.config import GuardConfig


def is_likely_free_text(profile: dict, guard: GuardConfig) -> bool:
    """
    基于画像统计信号判断是否为自由文本列。

    满足任一结构化条件则返回 False（跳过 guard）:
        - top5 覆盖率高且模式种类少
        - numeric_stats 非空（>=80% 可解析为数字）
        - 单一主导模式占比极高
    """
    pattern_coverage = profile.get("pattern_coverage") or {}        # 获取 pattern_coverage 信息，若无则用空字典
    top5_rate = pattern_coverage.get("top5_coverage_rate", 0)       # 获取 top5 覆盖率（未找到则默认为0）
    distinct_count = pattern_coverage.get("distinct_pattern_count", 999)  # 获取不同模式的个数（未找到则设置为很大）

    # 如果 top5 覆盖率很高 且 模式种类很少，则认为结构化字段，返回 False
    if (
        top5_rate >= guard.top5_coverage_structured_min
        and distinct_count <= guard.distinct_pattern_structured_max
    ):
        return False

    # 如果 numeric_stats 不为空，说明是数值型列，直接返回 False
    if profile.get("numeric_stats") is not None:
        return False

    patterns = profile.get("observed_patterns") or []           # 获取所有观测到的 pattern 及计数
    total = profile.get("total_count") or 0                     # 获取总记录条数
    # 如果有 pattern 且总数大于0，计算主导模式的占比
    if patterns and total > 0:
        dominant_rate = patterns[0][1] / total                 # 主导pattern出现的频率
        if dominant_rate >= guard.dominant_pattern_rate_min:   # 若主导模式极高，判为结构化
            return False

    # 否则判断为自由文本列
    return True


def check_regex_too_strict(pattern_str: str, profile: dict) -> Tuple[bool, List[str]]:
    """
    检查 regex 是否无法匹配 edge_samples 中的完整值。

    Returns:
        (too_strict, failing_values)
    """
    edge_samples = profile.get("edge_samples") or []      # 获取 edge_samples，如果没有则用空列表
    if not edge_samples:
        return False, []                                  # 没有样本，不判定严格，直接返回

    if not pattern_str:
        return True, edge_samples                         # 如果没有正则表达式，认为全部都匹配失败

    try:
        pattern = re.compile(pattern_str)                 # 尝试编译正则表达式
    except re.error:
        return True, ["<invalid_regex>"]                  # 如果正则语法错误，全部匹配失败

    # 对 edge_samples 中的每个值做 pattern.match，如果不匹配则加入 failing 列表
    failing = [str(v) for v in edge_samples if pattern.match(str(v)) is None]
    return bool(failing), failing                         # 只要有不匹配的，就认为过严，并返回不匹配的值


def should_drop_rule(
    rule: dict,
    profile: dict,
    column: str,
    guard: GuardConfig,
) -> bool:
    """判断是否应在编译前丢弃该规则。"""
    if not guard.enabled:                       # 如果 guard 没开启，直接返回 False
        return False
    if rule.get("type") != "regex":             # 如果规则类型不是 regex，直接返回 False
        return False
    if not is_likely_free_text(profile, guard): # 不属于自由文本列，直接返回 False
        return False

    spec = rule.get("spec") or {}               # 取出规则描述（spec），没有则用空字典
    pattern_str = spec.get("pattern")           # 获取正则 pattern 字符串
    too_strict, failing = check_regex_too_strict(pattern_str, profile) # 检查正则是否过严
    if not too_strict:                          # 没有过严则保留
        return False

    preview = failing[:3]                       # 只取前3个不通过的样本做日志预览
    print(
        f"[discard] 列 {column} regex 无法匹配 edge_samples {preview!r},丢弃 "
        f"(pattern: {pattern_str!r}, reason: {rule.get('reason', '')})"
    )   # 打印日志说明被丢弃的原因
    return True                                # 判定需要丢弃规则
