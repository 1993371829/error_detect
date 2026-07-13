"""
近似函数依赖(FD)挖掘 -> 违反依赖错误(VAD)。

规则层是纯逐列的，无法发现"跨列依赖违反"。本模块从数据中自动发现
近似函数依赖 A -> B（如 ZipCode -> City / State, MeasureCode -> MeasureName），
再把违反"多数派映射"的行标记为 VAD。

判定流程:
    1. 枚举有序列对 (A, B)。
    2. 对每个 A 取值，统计其对应 B 值的分布，得到主导 B 值及其占比。
    3. 用支持度加权的整体一致率衡量 FD 是否近似成立(>= min_confidence)。
    4. 对成立的 FD，在"组足够大且主导占比足够高"的 A 组里，
       把 B != 主导值 的行标为 VAD，suggested_fix = 主导 B 值。

仅做统计，不调用 LLM。空值不参与 FD 统计与违反判定。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd

from stage_1.profiling import is_blank


class FunctionalDependency:
    """一条近似函数依赖 A -> B 及其每个 A 值对应的主导 B 值。"""

    def __init__(
        self,
        determinant: str,
        dependent: str,
        confidence: float,
        mapping: dict[str, tuple[str, float, int]],
    ):
        self.determinant = determinant      # A 列名
        self.dependent = dependent          # B 列名
        self.confidence = confidence        # 全局一致率
        # a_value -> (主导 B 值, 组内主导占比, 组大小)
        self.mapping = mapping
        self.semantic_reason = ""           # LLM 语义校验给出的理由（若启用）

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FD({self.determinant} -> {self.dependent}, "
            f"conf={self.confidence:.3f}, groups={len(self.mapping)})"
        )


def _column_cardinality(series: pd.Series) -> int:
    """非空唯一值个数。"""
    non_blank = series[~series.map(is_blank)].astype(str)
    return non_blank.nunique()


def discover_fd(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    *,
    min_confidence: float = 0.9,
    min_group_support: int = 5,
    min_group_confidence: float = 0.9,
    pure_threshold: float = 0.9,
    min_pure_group_ratio: float = 0.85,
    min_distinct_dependents: int = 2,
) -> Optional[FunctionalDependency]:
    """
    尝试发现近似 FD a_col -> b_col。

    判定（三重门槛，兼顾精度）:
        1. 全局一致率 >= min_confidence。
        2. "纯组比例" >= min_pure_group_ratio：在有足够支持度的 A 组里，
           主导占比 >= pure_threshold 的组所占比例。
           - 真·键依赖(如 ProviderNumber->State)绝大多数组是纯的(一个 A 对一个 B)，
             少数不纯是注入错误所致 -> 纯组比例高，保留。
           - 软相关(如 CountyName 跨多州)大量组天然不纯 -> 纯组比例低，丢弃。
        3. 不同主导值数 >= min_distinct_dependents：mapping 中主导 B 值的去重个数。
           - 防"类别不平衡假依赖"：当 B 极度倾斜(如 State 98% 为 'al')时，
             任意 A(如 EmergencyService) 都会"预测"多数类，confidence 虚高且
             所有组主导值都相同。要求 A 能解释 B 的多样性(>=2 个不同主导值)，
             才认为是真依赖。

    Returns:
        通过三重门槛则返回 FunctionalDependency；mapping 仅含
        "组大小 >= min_group_support 且主导占比 >= min_group_confidence" 的 A 组
        （这些组才用于后续违反判定）。否则返回 None。
    """
    groups: dict[str, "defaultdict[str, int]"] = defaultdict(lambda: defaultdict(int))
    total = 0
    for a_val, b_val in zip(df[a_col], df[b_col]):
        if is_blank(a_val) or is_blank(b_val):
            continue
        groups[str(a_val)][str(b_val)] += 1
        total += 1

    if total == 0 or not groups:
        return None

    agreement = 0
    mapping: dict[str, tuple[str, float, int]] = {}
    supported_groups = 0
    pure_groups = 0
    for a_val, b_counts in groups.items():
        group_size = sum(b_counts.values())
        dom_b, dom_count = max(b_counts.items(), key=lambda kv: kv[1])
        dom_share = dom_count / group_size
        agreement += dom_count
        if group_size >= min_group_support:
            supported_groups += 1
            if dom_share >= pure_threshold:
                pure_groups += 1
            if dom_share >= min_group_confidence:
                mapping[a_val] = (dom_b, dom_share, group_size)

    confidence = agreement / total
    if confidence < min_confidence or not mapping:
        return None
    if supported_groups == 0:
        return None
    pure_ratio = pure_groups / supported_groups
    if pure_ratio < min_pure_group_ratio:
        return None
    distinct_dependents = len({dom_b for dom_b, _, _ in mapping.values()})
    if distinct_dependents < min_distinct_dependents:
        return None
    return FunctionalDependency(a_col, b_col, confidence, mapping)


def discover_all_fds(
    df: pd.DataFrame,
    *,
    min_confidence: float = 0.9,
    min_group_support: int = 5,
    min_group_confidence: float = 0.9,
    pure_threshold: float = 0.9,
    min_pure_group_ratio: float = 0.85,
    min_distinct_dependents: int = 2,
    max_determinant_unique_ratio: float = 0.5,
    min_dependent_unique: int = 2,
) -> list[FunctionalDependency]:
    """
    在所有有序列对上挖掘近似 FD。

    剪枝:
        - 跳过近似唯一的决定列(unique/total > max_determinant_unique_ratio):
          这类列(如行级 id)FD 平凡成立且不产生有意义的违反。
        - 跳过常量依赖列(unique < min_dependent_unique)。
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    cardinalities = {col: _column_cardinality(df[col]) for col in df.columns}

    fds: list[FunctionalDependency] = []
    for a_col in df.columns:
        a_card = cardinalities[a_col]
        if a_card <= 1:
            continue
        if a_card / n_rows > max_determinant_unique_ratio:
            continue
        for b_col in df.columns:
            if a_col == b_col:
                continue
            if cardinalities[b_col] < min_dependent_unique:
                continue
            fd = discover_fd(
                df, a_col, b_col,
                min_confidence=min_confidence,
                min_group_support=min_group_support,
                min_group_confidence=min_group_confidence,
                pure_threshold=pure_threshold,
                min_pure_group_ratio=min_pure_group_ratio,
                min_distinct_dependents=min_distinct_dependents,
            )
            if fd is not None:
                fds.append(fd)
    return fds


def detect_fd_violations(
    df: pd.DataFrame,
    fds: list[FunctionalDependency],
) -> list[dict]:
    """
    根据已发现的 FD 列表，标记违反主导映射的单元格为 VAD。

    被标记的是"依赖列(B)"的单元格，suggested_fix 为该 A 值对应的主导 B 值。
    confidence 取组内主导占比。
    """
    errors: list[dict] = []
    for fd in fds:
        a_col, b_col = fd.determinant, fd.dependent
        for row_id, (a_val, b_val) in enumerate(zip(df[a_col], df[b_col])):
            if is_blank(a_val) or is_blank(b_val):
                continue
            entry = fd.mapping.get(str(a_val))
            if entry is None:
                continue
            dom_b, dom_share, group_size = entry
            if str(b_val) == dom_b:
                continue
            # df 的索引与位置一致（read_csv 默认 RangeIndex），用 index 取真实 row_id
            real_row_id = df.index[row_id]
            errors.append({
                "row_id": real_row_id,
                "column": b_col,
                "value": b_val,
                "error_type": "VAD",
                "violated_rule": f"fd:{a_col}->{b_col}",
                "reason": (
                    f"违反依赖 {a_col}->{b_col}: 当 {a_col}='{a_val}' 时 "
                    f"{b_col} 多数为 '{dom_b}'({dom_share:.0%}, n={group_size}), "
                    f"但此处为 '{b_val}'"
                ),
                "suggested_fix": dom_b,
                "confidence": round(dom_share, 3),
            })
    return errors


# --------------------------------------------------------------------------- #
# 分档验证支持（二期）：子样本一致率重算 + 三档 LLM 审核 prompt
# --------------------------------------------------------------------------- #

def fd_confidence_on(df: pd.DataFrame, a_col: str, b_col: str, idx) -> Optional[float]:
    """在给定行位置子集上重算 A->B 的全局一致率（供 bootstrap 稳定性用）。"""
    a_vals = df[a_col].to_numpy()
    b_vals = df[b_col].to_numpy()
    groups: dict[str, "defaultdict[str, int]"] = defaultdict(lambda: defaultdict(int))
    total = 0
    for pos in idx:
        av, bv = a_vals[pos], b_vals[pos]
        if is_blank(av) or is_blank(bv):
            continue
        groups[str(av)][str(bv)] += 1
        total += 1
    if total == 0:
        return None
    agreement = sum(max(counts.values()) for counts in groups.values())
    return agreement / total


FD_AUDIT_PROMPT_GRADED = """你是数据质量专家。下面是从一张表中统计挖掘出的"候选函数依赖" A -> B，
即 A 列的取值疑似能决定 B 列的取值。请判断该依赖在现实语义上的可信程度。

候选依赖: {a} -> {b}
样例映射（A 值 => B 的多数值，占比，组大小）:
{samples}

判断标准与三档输出:
- high  : A 在概念上确实决定 B（如 ZipCode->City/State、ProviderNumber->HospitalName、
          MeasureCode->MeasureName），该依赖真实成立，违反者应视为错误。
- medium: 可能成立但拿不准（弱相关/部分成立/样例不足以确认），保留为弱证据但不作确定性结论。
- drop  : 伪依赖（A 与 B 无现实因果/标识关系，仅因某列取值高度集中而"碰巧"一致）。

只输出 JSON（不要任何解释）:
{{"tier": "high|medium|drop", "reason": "简短理由"}}"""


def detect_vad_graded(
    df: pd.DataFrame,
    *,
    thresholds,
    min_confidence: float = 0.9,
    min_group_support: int = 5,
    min_group_confidence: float = 0.9,
    pure_threshold: float = 0.9,
    min_pure_group_ratio: float = 0.85,
    min_distinct_dependents: int = 2,
    max_determinant_unique_ratio: float = 0.5,
    min_dependent_unique: int = 2,
    llm=None,
    cache=None,
) -> list[dict]:
    """
    分档版 FD 检测（二期）：挖掘 FD 后逐条经统一分档验证赋 severity，再生成 VAD 错误。

    每条 FD:
        统计强 -> high（进 mask）；统计弱 -> drop；灰区 -> LLM 三档审核（high/medium/drop）。
    产出的每个错误 dict 带 `severity` 字段，drop 的 FD 不产生错误。
    """
    from stage_1.rule_validation import grade_rule

    fds = discover_all_fds(
        df,
        min_confidence=min_confidence,
        min_group_support=min_group_support,
        min_group_confidence=min_group_confidence,
        pure_threshold=pure_threshold,
        min_pure_group_ratio=min_pure_group_ratio,
        min_distinct_dependents=min_distinct_dependents,
        max_determinant_unique_ratio=max_determinant_unique_ratio,
        min_dependent_unique=min_dependent_unique,
    )
    n_rows = len(df)
    kept_fds: list[FunctionalDependency] = []
    severities: dict[tuple[str, str], str] = {}
    n_high = n_med = n_drop = 0
    for fd in fds:
        # 该 FD 的 support = 参与统计的非空行数（近似用组大小之和）
        support = sum(size for _, _, size in fd.mapping.values())
        samples = "\n".join(
            f"  {a!r} => {dom!r} ({share:.0%}, n={size})"
            for a, (dom, share, size) in list(fd.mapping.items())[:15]
        )
        prompt = FD_AUDIT_PROMPT_GRADED.format(a=fd.determinant, b=fd.dependent, samples=samples)
        audit_profile = {
            "_task": "fd_audit",
            "a": fd.determinant, "b": fd.dependent,
            "samples": samples,
        }
        severity, stability, reason = grade_rule(
            support=support,
            confidence=fd.confidence,
            n_rows=n_rows,
            th=thresholds,
            recompute_confidence=lambda idx, a=fd.determinant, b=fd.dependent: fd_confidence_on(df, a, b, idx),
            llm=llm, cache=cache,
            audit_profile=audit_profile, audit_prompt=prompt,
            label=f"FD {fd.determinant}->{fd.dependent}",
        )
        fd.semantic_reason = reason
        if severity == "drop":
            n_drop += 1
            print(f"[fd-drop] {fd.determinant}->{fd.dependent}: {reason}")
            continue
        severities[(fd.determinant, fd.dependent)] = severity
        kept_fds.append(fd)
        if severity == "high":
            n_high += 1
        else:
            n_med += 1
    print(f"FD 分档: {len(fds)} 候选 -> high {n_high} / medium {n_med} / drop {n_drop}")

    errors = detect_fd_violations(df, kept_fds)
    for err in errors:
        vr = str(err.get("violated_rule", ""))  # 形如 fd:A->B
        a_b = vr[3:].split("->", 1) if vr.startswith("fd:") else None
        sev = severities.get((a_b[0], a_b[1]), "medium") if a_b and len(a_b) == 2 else "medium"
        err["severity"] = sev
    return errors
