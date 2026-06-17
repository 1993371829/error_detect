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

from stage_1.llm_rules import complete_json
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
# FD 语义校验（借鉴 Cocoon：统计挖掘出强 FD 后，让 LLM 判断其是否语义上真实成立）
# --------------------------------------------------------------------------- #

FD_SEMANTIC_PROMPT = """你是数据质量专家。下面是从一张表中统计挖掘出的"候选函数依赖" A -> B，
即 A 列的取值疑似能决定 B 列的取值。请判断该依赖在现实语义上是否真实成立
（即 A 在概念上确实决定 B，而非因数据分布巧合/类别不平衡造成的伪相关）。

候选依赖: {a} -> {b}
样例映射（A 值 => B 的多数值，占比，组大小）:
{samples}

判断标准:
- 真实依赖示例: ZipCode -> City/State、ProviderNumber -> HospitalName、MeasureCode -> MeasureName。
- 伪依赖示例: A 与 B 无现实因果/标识关系，仅因某列取值高度集中而"碰巧"一致。

只输出 JSON（不要任何解释）:
{{"valid": true/false, "reason": "简短理由"}}"""


def is_fd_semantically_valid(fd: FunctionalDependency, llm, max_samples: int = 15) -> tuple[bool, str]:
    """
    用 LLM 判断候选 FD 是否语义上真实成立。

    解析/调用失败时保守保留（返回 True），避免因校验环节异常而误删真实错误。
    """
    sample_lines = []
    for a_val, (dom_b, share, size) in list(fd.mapping.items())[:max_samples]:
        sample_lines.append(f"  {a_val!r} => {dom_b!r} ({share:.0%}, n={size})")
    prompt = FD_SEMANTIC_PROMPT.format(
        a=fd.determinant, b=fd.dependent, samples="\n".join(sample_lines),
    )
    # 复用规则归纳同款健壮解析（max_tokens 截断保护 + 宽松解析 + 重试）
    data = complete_json(llm, prompt, label=f"FD {fd.determinant}->{fd.dependent}")
    if isinstance(data, dict):
        return bool(data.get("valid", True)), str(data.get("reason", ""))
    print(f"[fd-warn] {fd.determinant}->{fd.dependent} 语义校验失败，保守保留")
    return True, "语义校验失败，保守保留"


def filter_fds_semantically(fds: list[FunctionalDependency], llm) -> list[FunctionalDependency]:
    """对候选 FD 逐条做 LLM 语义校验，仅保留被确认为真实成立的 FD。"""
    if not fds or llm is None:
        return fds
    kept: list[FunctionalDependency] = []
    for fd in fds:
        valid, reason = is_fd_semantically_valid(fd, llm)
        fd.semantic_reason = reason
        if valid:
            kept.append(fd)
        else:
            print(f"[fd-drop] 语义校验否决 {fd.determinant}->{fd.dependent}: {reason}")
    return kept


def detect_vad(
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
    llm=None,
    semantic_check: bool = False,
) -> tuple[list[dict], list[FunctionalDependency]]:
    """
    挖掘 FD 并返回 (VAD 错误记录, 发现的 FD 列表)。

    当 semantic_check=True 且提供 llm 时，在统计挖掘后增加一步 LLM 语义校验，
    只有被确认语义成立的 FD 才用于生成 VAD 错误（借鉴 Cocoon，降低伪依赖误报）。
    """
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
    if semantic_check and llm is not None and fds:
        before = len(fds)
        fds = filter_fds_semantically(fds, llm)
        print(f"FD 语义校验: {before} 条候选 -> 保留 {len(fds)} 条")
    errors = detect_fd_violations(df, fds)
    return errors, fds
