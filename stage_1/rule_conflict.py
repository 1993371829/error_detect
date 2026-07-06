"""
规则冲突消解（二期）——多规则族（FD/CFD/DC）共存时的必需项。

当 FD/CFD/DC 对**同一 target 单元格**给出结论（尤其 suggested_fix 不一致）时，
按优先级择一保留，其余丢弃；若无法判定明确胜者，则把保留者降为 medium
（不进 clean_mask），避免用相互矛盾、不确定的规则污染 Stage 2 训练分布。

优先级（从高到低）:
    1. 用户规则     (来源标记，当前默认无)
    2. confidence   (规则一致率 / 组内主导占比)
    3. 规则特异性   (CFD 条件更具体 > DC 结构约束 > 全局 FD)
    4. severity     (high > medium)

在 executor 汇总所有规则族错误后、build_clean_mask 前调用。
"""

from __future__ import annotations

from collections import defaultdict


_SPECIFICITY = {"cfd": 3, "dc": 2, "fd": 1}


def _family(err: dict) -> str:
    vr = str(err.get("violated_rule", ""))
    if vr.startswith("cfd:"):
        return "cfd"
    if vr.startswith("dc:"):
        return "dc"
    if vr.startswith("fd:") or err.get("error_type") == "VAD":
        return "fd"
    return "other"


def _priority_key(err: dict, user_rule_prefixes: tuple[str, ...]) -> tuple:
    vr = str(err.get("violated_rule", ""))
    is_user = 1 if any(vr.startswith(p) for p in user_rule_prefixes) else 0
    conf = float(err.get("confidence") or 0.0)
    spec = _SPECIFICITY.get(_family(err), 0)
    sev = 1 if str(err.get("severity", "")).lower() == "high" else 0
    return (is_user, conf, spec, sev)


def resolve_rule_conflicts(
    rule_errors: list[dict],
    *,
    user_rule_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """
    对规则族错误按 (row_id, column) 去冲突，返回每个单元格保留的唯一记录。

    - 单条：原样保留。
    - 多条且 suggested_fix 冲突、且首名与次名优先级相同（无法判定）：保留首名但降 medium。
    - 多条：保留优先级最高者，其余丢弃。
    """
    if not rule_errors:
        return []

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for err in rule_errors:
        groups[(err.get("row_id"), err.get("column"))].append(err)

    resolved: list[dict] = []
    n_conflict = n_downgrade = 0
    for _cell, records in groups.items():
        if len(records) == 1:
            resolved.append(records[0])
            continue

        ranked = sorted(records, key=lambda e: _priority_key(e, user_rule_prefixes), reverse=True)
        winner = dict(ranked[0])

        fixes = {
            str(r.get("suggested_fix")).strip()
            for r in records
            if r.get("suggested_fix") not in (None, "", "nan")
        }
        has_conflict = len(fixes) >= 2
        if has_conflict:
            n_conflict += 1
            top_key = _priority_key(ranked[0], user_rule_prefixes)
            second_key = _priority_key(ranked[1], user_rule_prefixes)
            if top_key == second_key:
                # 无法判定明确胜者 -> 降 medium（不进 mask）
                if str(winner.get("severity", "")).lower() == "high":
                    n_downgrade += 1
                winner["severity"] = "medium"
                winner["reason"] = (
                    str(winner.get("reason", "")) + " [冲突消解: 多规则结论不一致且优先级相当，降级 medium]"
                )
        resolved.append(winner)

    if n_conflict:
        print(f"规则冲突消解: {n_conflict} 个单元格存在冲突，其中 {n_downgrade} 个降级为 medium")
    return resolved
