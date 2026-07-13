"""
统一规则分档验证骨架（FD / CFD / DC 共用）。

对每条候选规则计算 support / confidence / bootstrap_stability，并据此分流：
    - 统计强（高支持 + 高一致率 + 高稳定）        -> high（跳过 LLM，省 token）
    - 统计弱（支持不足 或 一致率过低）            -> drop
    - 灰区                                        -> 调 LLM 三档审核（high/medium/drop）

severity 语义与一期双轨一致：
    - high   : 进 clean_mask（净化训练分布）
    - medium : 仅作弱证据，不进 mask（避免用不确定规则污染 Stage 2 训练集）
    - drop   : 丢弃，不产生错误

LLM 审核结果写入 rule_cache（复现 + 省钱）；无 LLM 时灰区保守降为 medium（不误删、不污染）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

TIER_VALUES = {"high", "medium", "drop"}


@dataclass
class GradeThresholds:
    """规则分档阈值（FD/CFD/DC 共用）。"""

    # 统计强 -> 直接 high
    high_min_support: int = 50
    high_min_confidence: float = 0.95
    high_min_stability: float = 0.90
    # 统计弱 -> 直接 drop
    drop_max_support: int = 10
    drop_max_confidence: float = 0.80
    # bootstrap 稳定性
    bootstrap_rounds: int = 30
    bootstrap_ratio: float = 0.80
    stability_conf_floor: float = 0.90   # 子样本内一致率达标视为"稳定命中"


def bootstrap_stability(
    recompute_confidence: Optional[Callable[[np.ndarray], Optional[float]]],
    n_rows: int,
    th: GradeThresholds,
    *,
    seed: int = 0,
) -> float:
    """
    自助采样稳定性：多轮 80% 无放回重采样后重算一致率，达标轮次占比。

    recompute_confidence(idx) 接收行位置数组，返回该子样本上的规则一致率（None 表示不可算）。
    为 None 时返回 1.0（视为稳定，交由统计/LLM 其余判据把关）。
    """
    if recompute_confidence is None:
        return 1.0
    if n_rows <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    m = max(1, int(th.bootstrap_ratio * n_rows))
    if m >= n_rows:
        m = n_rows - 1
    hits = 0
    for _ in range(th.bootstrap_rounds):
        idx = rng.choice(n_rows, size=m, replace=False)
        conf = recompute_confidence(idx)
        if conf is not None and conf >= th.stability_conf_floor:
            hits += 1
    return hits / th.bootstrap_rounds


def _parse_tier(data) -> tuple[str, str]:
    """解析 LLM 审核返回，映射为 (tier, reason)。解析失败保守 medium。"""
    if not isinstance(data, dict):
        return "medium", "审核解析失败，保守 medium"
    t = str(data.get("tier", "")).strip().lower()
    if t in TIER_VALUES:
        return t, str(data.get("reason", ""))
    # 兼容旧版二元 valid true/false
    if "valid" in data:
        return ("high" if bool(data["valid"]) else "drop"), str(data.get("reason", ""))
    return "medium", "审核字段缺失，保守 medium"


def graded_llm_audit(
    audit_profile: dict,
    prompt: str,
    llm,
    cache,
    *,
    label: str = "",
) -> tuple[str, str]:
    """
    灰区规则的 LLM 三档审核：返回 (tier, reason)。

    audit_profile: 供 rule_cache 索引的 dict（需含 `_task` 命名空间键，避免与列规则缓存冲突）。
    prompt: 要求输出 {"tier": "high|medium|drop", "reason": "..."} 的审核提示词。
    命中缓存直接返回；llm 为空时保守返回 ('medium', ...)。
    """
    if cache is not None:
        cached = cache.get(audit_profile)
        if isinstance(cached, dict) and cached.get("tier") in TIER_VALUES:
            return cached["tier"], cached.get("reason", "")
    if llm is None:
        return "medium", "无 LLM，灰区规则保守降级为 medium"
    from stage_1.llm_rules import complete_json

    data = complete_json(llm, prompt, label=label)
    tier, reason = _parse_tier(data)
    if cache is not None:
        cache.set(audit_profile, {"tier": tier, "reason": reason})
    return tier, reason


def grade_rule(
    *,
    support: int,
    confidence: float,
    n_rows: int,
    th: GradeThresholds,
    recompute_confidence: Optional[Callable[[np.ndarray], Optional[float]]] = None,
    llm=None,
    cache=None,
    audit_profile: Optional[dict] = None,
    audit_prompt: Optional[str] = None,
    label: str = "",
    seed: int = 0,
) -> tuple[str, float, str]:
    """
    综合分档：统计预分流 -> 灰区 LLM 审核。

    Returns:
        (severity, stability, reason)，severity ∈ {high, medium, drop}。
    """
    # 粗筛：drop 与"不可能 high"仅由 support/confidence 决定，跳过 30 轮 bootstrap
    # （stability 只影响统计强直通分支，判定结果与全量计算完全一致）。
    if support < th.drop_max_support or confidence < th.drop_max_confidence:
        return "drop", 0.0, f"统计弱(support={support},conf={confidence:.2f})"
    if support >= th.high_min_support and confidence >= th.high_min_confidence:
        stability = bootstrap_stability(recompute_confidence, n_rows, th, seed=seed)
        if stability >= th.high_min_stability:
            return "high", stability, f"统计强(support={support},conf={confidence:.2f},stab={stability:.2f})"
    else:
        stability = 0.0
    # 灰区 -> LLM 三档审核
    if audit_prompt is None:
        return "medium", stability, "灰区无审核 prompt，保守 medium"
    profile = audit_profile or {"_task": "rule_audit", "label": label}
    sev, reason = graded_llm_audit(profile, audit_prompt, llm, cache, label=label)
    return sev, stability, reason
