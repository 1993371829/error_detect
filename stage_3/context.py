"""
Stage 3 上下文构造：把候选错误按行分组，并附带整行值、同列正常样例、列语义类型，
以及——针对"跨源/跨行共识型"列——同 key 分组的多数值证据，供 prompt 构造使用。

设计要点（解决 flights 这类"多源时刻冲突"在精检阶段被误杀的问题）:
    - flights 的错误（同一 flight 被不同 src 报告的时刻不一致）无法靠单行判定，
      必须对比同一 key（flight）下其他行的取值。
    - 这里自动探测 key 列，并为每个可疑格计算"同 key 多数值/占比/本值占比/是否冲突"，
      写入上下文供 LLM 判断；同时给 verifier 提供"是否与多数值冲突"的保护信号。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.io_utils import read_table

# 高置信确定性结构错误：这类前序判定无需 LLM 复核，直接确认（省 LLM 调用、防误杀）。
# - duplicate_value：整值由同一 token 重复拼接，Stage1 实测精度 0.99+。
# - format_outlier：格式高度统一的列里偏离主导形态的值（双门控），实测精度 0.98+。
AUTO_CONFIRM_RULES = {"duplicate_value", "format_outlier"}


@dataclass
class SuspectCell:
    """一个可疑单元格及其来自前序阶段的线索。"""

    column: str
    value: str
    prior_error_type: str            # 前序阶段判定的类型（FI/MV/T/VAD/DIST）
    prior_source: str                # stage1 / stage2
    reason: str                      # 前序原因说明
    suggested_fix: str               # 前序建议修复（可能为空）
    semantic_type: str               # 该列语义类型（来自 rules.json）
    normal_samples: list = field(default_factory=list)  # 该列正常高频样例
    anomaly_score: Optional[float] = None  # Stage2 分布模型异常分（stage1 候选为 None）
    subtype: str = ""                # Stage2 子类型（categorical/numeric/surrogate）
    verifiability: str = "verifiable"  # verifiable / consensus_only
    consensus: Optional[dict] = None   # 同 key 共识证据（见 _cell_consensus）
    column_stats: Optional[dict] = None  # 该列统计画像（借鉴 Cocoon，供 LLM 基于分布判断）
    auto_confirm: bool = False         # 高置信确定性结构错误，直通确认不经 LLM

    @property
    def consensus_conflict(self) -> bool:
        """本格取值与同 key 多数值真正冲突（跨行证据支持其为错误）。"""
        return bool(self.consensus and self.consensus.get("is_conflict"))


@dataclass
class RowContext:
    """一行的完整上下文：整行值 + 该行所有可疑单元格。"""

    row_id: int
    row_values: dict                 # 列 -> 值（整行，用于跨列判断）
    suspects: list                   # list[SuspectCell]


def load_semantic_types(rules_path: str | Path) -> dict:
    """从 Stage1 rules.json 读取 列 -> semantic_type 映射（跳过 FD 汇总条目）。"""
    path = Path(rules_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    out = {}
    for entry in rules:
        if isinstance(entry, dict) and "column" in entry:
            out[entry["column"]] = entry.get("semantic_type", "")
    return out


def compute_normal_samples(df: pd.DataFrame, max_samples: int = 8) -> dict:
    """每列取出现频次最高的若干非空值作为"正常样例"。"""
    out = {}
    for col in df.columns:
        series = df[col]
        non_blank = series[~series.map(is_blank)].astype(str)
        if non_blank.empty:
            out[col] = []
        else:
            out[col] = non_blank.value_counts().head(max_samples).index.tolist()
    return out


def _to_pattern(s: str) -> str:
    """将值抽象为字符模式（数字->d, 大写->L, 小写->l），用于呈现该列的主导格式。"""
    out = []
    for c in s:
        if c.isdigit():
            out.append("d")
        elif c.isupper():
            out.append("L")
        elif c.islower():
            out.append("l")
        else:
            out.append(c)
    return "".join(out)


def compute_column_stats(df: pd.DataFrame, max_top: int = 8) -> dict:
    """
    借鉴 Cocoon：为每列计算轻量统计画像，供 Stage 3 prompt 注入，帮助 LLM 基于分布
    （而非仅单值）判断 string/pattern/numeric outlier 与伪缺失值。

    返回 col -> {
        null_rate, distinct_count, total,
        top_values: [[value, count], ...]（频次降序，含占比信息由 count/total 推得）,
        dominant_patterns: [模式, ...]（字符形态 top 3）,
        numeric_range: {min, max, mean} 或 None（仅当 >=80% 可解析为数值）,
    }
    """
    out: dict = {}
    total = len(df)
    for col in df.columns:
        series = df[col]
        blank_mask = series.map(is_blank)
        non_blank = series[~blank_mask].astype(str)
        null_rate = round(float(blank_mask.sum()) / total, 3) if total else 0.0
        if non_blank.empty:
            out[str(col)] = {
                "null_rate": null_rate,
                "distinct_count": 0,
                "total": total,
                "top_values": [],
                "dominant_patterns": [],
                "numeric_range": None,
            }
            continue

        vc = non_blank.value_counts()
        top_values = [[str(v), int(c)] for v, c in vc.head(max_top).items()]

        patterns = non_blank.map(_to_pattern).value_counts().head(3)
        dominant_patterns = [str(p) for p in patterns.index.tolist()]

        nums = pd.to_numeric(non_blank, errors="coerce").dropna()
        if len(nums) >= 0.8 * len(non_blank) and len(nums) > 0:
            numeric_range = {
                "min": float(nums.min()),
                "max": float(nums.max()),
                "mean": round(float(nums.mean()), 2),
            }
        else:
            numeric_range = None

        out[str(col)] = {
            "null_rate": null_rate,
            "distinct_count": int(non_blank.nunique()),
            "total": total,
            "top_values": top_values,
            "dominant_patterns": dominant_patterns,
            "numeric_range": numeric_range,
        }
    return out


# --------------------------------------------------------------------------- #
# 跨行/跨源共识证据
# --------------------------------------------------------------------------- #

def detect_key_columns(df: pd.DataFrame, min_avg_group: float = 3.0) -> list[str]:
    """
    探测可作为分组 key 的列：同一取值在多行中重复出现（平均每个取值 >= min_avg_group 行），
    且至少有 2 个不同取值。flights 中 flight / src 会被识别。
    """
    keys = []
    for col in df.columns:
        nb = df[col][~df[col].map(is_blank)].astype(str)
        nu = nb.nunique()
        if nu < 2:
            continue
        if len(nb) / nu >= min_avg_group:
            keys.append(col)
    return keys


def _column_consensus(
    df: pd.DataFrame, col: str, key: str, min_cell_group: int = 3,
) -> tuple[float, int, dict]:
    """
    以 key 分组，统计 col 在每组的取值分布与主导值。
    主导占比只在"足够大的组"（size >= min_cell_group）上加权计算，避免高基数 key
    用大量小组过拟合出虚高的占比。
    返回 (大组加权平均主导占比, 组数, {key_value: {vc, size, majority, majority_share}})。
    """
    sub = pd.DataFrame({
        "k": df[key].map(lambda v: "" if is_blank(v) else str(v)),
        "v": df[col].map(lambda v: "" if is_blank(v) else str(v)),
    })
    groups: dict = {}
    big_w = 0.0
    big_dom = 0.0
    for kval, g in sub.groupby("k", sort=False):
        if kval == "":
            continue
        vals = g["v"][g["v"] != ""]
        size = int(len(vals))
        if size == 0:
            continue
        vc = vals.value_counts()
        maj = str(vc.index[0])
        maj_share = float(vc.iloc[0]) / size
        groups[str(kval)] = {
            "vc": vc, "size": size, "majority": maj, "majority_share": maj_share,
        }
        if size >= min_cell_group:
            big_w += size
            big_dom += maj_share * size
    dom_big = (big_dom / big_w) if big_w else 0.0
    return dom_big, len(groups), groups


def _global_top_share(df: pd.DataFrame, col: str) -> tuple[float, int]:
    """该列全局最高频非空值的占比与去重值数。"""
    nb = df[col][~df[col].map(is_blank)].astype(str)
    n = len(nb)
    if n == 0:
        return 0.0, 0
    vc = nb.value_counts()
    return float(vc.iloc[0]) / n, int(vc.shape[0])


def compute_consensus(
    df: pd.DataFrame,
    candidate_columns,
    *,
    min_avg_group: float = 3.0,
    min_dominance: float = 0.5,
    min_cell_group: int = 3,
    min_lift: float = 0.15,
) -> dict:
    """
    为每个候选列挑选最合适的 key 列形成共识，返回 col -> {key_column, dominance, groups}。

    选择准则（关键）:
        1. dom_big >= min_dominance：key 在大组上确实能主导地预测该列。
        2. lift = dom_big - 全局最高占比 >= min_lift：分组必须显著降低不确定性，
           否则只是该列本身类别不平衡（如某列 95% 同值）冒充的伪共识，予以排除。
        3. 决定列必须比依赖列"更粗"（去重值不多于它），排除高基数过拟合，并使实体标识列
           （如 flight）拿不到 key、其误报仍能被正常否决。
        4. 在合格 key 中优先选最粗的（分组数最少 = 重复最多的实体标识），同粗则取占比更高者。
    若没有任何 key 合格，则该列不收录（不提供跨行共识证据，Stage 3 行为与改造前一致）。
    """
    keys = detect_key_columns(df, min_avg_group)
    result: dict = {}
    for col in candidate_columns:
        if col not in df.columns:
            continue
        global_top, col_nunique = _global_top_share(df, col)
        best = None  # (n_groups, -dom_big, key, dom_big, groups)
        for key in keys:
            if key == col:
                continue
            dom_big, n_groups, groups = _column_consensus(df, col, key, min_cell_group)
            if dom_big < min_dominance:
                continue
            if (dom_big - global_top) < min_lift:
                continue
            if n_groups > col_nunique:
                continue
            cand = (n_groups, -dom_big, key, dom_big, groups)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is not None:
            result[col] = {
                "key_column": best[2],
                "dominance": round(best[3], 3),
                "groups": best[4],
            }
    return result


def _cell_consensus(
    consensus_map: dict,
    col: str,
    key_value,
    current_value,
    *,
    min_cell_group: int = 3,
    conflict_majority_share: float = 0.4,
    conflict_current_share_max: float = 0.5,
) -> Optional[dict]:
    """为单个可疑格计算同 key 共识证据；组太小或无证据时返回 None。"""
    info = consensus_map.get(col)
    if not info:
        return None
    grp = info["groups"].get(str(key_value))
    if not grp or grp["size"] < min_cell_group:
        return None
    vc = grp["vc"]
    cur = "" if is_blank(current_value) else str(current_value)
    current_count = int(vc.get(cur, 0))
    current_share = current_count / grp["size"]
    majority = grp["majority"]
    is_conflict = (
        cur != majority
        and grp["majority_share"] >= conflict_majority_share
        and current_share < conflict_current_share_max
    )
    return {
        "key_column": info["key_column"],
        "key_value": str(key_value),
        "group_size": grp["size"],
        "majority_value": majority,
        "majority_share": round(grp["majority_share"], 3),
        "current_share": round(current_share, 3),
        "is_conflict": bool(is_conflict),
    }


def _to_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def build_row_contexts(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    semantic_types: dict,
    normal_samples: dict,
    consensus_map: Optional[dict] = None,
    column_stats: Optional[dict] = None,
) -> list[RowContext]:
    """按 row_id 分组候选，组装 RowContext 列表（按 row_id 升序）。"""
    consensus_map = consensus_map or {}
    column_stats = column_stats or {}
    contexts: list[RowContext] = []
    candidates = candidates.copy()
    candidates["row_id"] = candidates["row_id"].astype(int)
    has_score = "anomaly_score" in candidates.columns
    has_subtype = "subtype" in candidates.columns

    for row_id, group in candidates.groupby("row_id", sort=True):
        if row_id < 0 or row_id >= len(df):
            continue
        row_values = {col: df.iloc[row_id][col] for col in df.columns}
        suspects = []
        for _, r in group.iterrows():
            col = str(r["column"])
            value = "" if pd.isna(r.get("value")) else str(r.get("value", ""))
            vr = r.get("violated_rule", "")
            violated_rule = "" if pd.isna(vr) else str(vr)
            consensus = None
            if col in consensus_map:
                key_col = consensus_map[col]["key_column"]
                consensus = _cell_consensus(
                    consensus_map, col, row_values.get(key_col, ""), value,
                )
            suspects.append(SuspectCell(
                column=col,
                value=value,
                prior_error_type=str(r.get("error_type", "") or ""),
                prior_source=str(r.get("source", "") or ""),
                reason=str(r.get("reason", "") or ""),
                suggested_fix="" if pd.isna(r.get("suggested_fix")) else str(r.get("suggested_fix", "") or ""),
                semantic_type=semantic_types.get(col, ""),
                normal_samples=normal_samples.get(col, []),
                anomaly_score=_to_float(r.get("anomaly_score")) if has_score else None,
                subtype=str(r.get("subtype", "") or "") if has_subtype else "",
                verifiability="consensus_only" if col in consensus_map else "verifiable",
                consensus=consensus,
                column_stats=column_stats.get(col),
                auto_confirm=violated_rule in AUTO_CONFIRM_RULES,
            ))
        contexts.append(RowContext(row_id=int(row_id), row_values=row_values, suspects=suspects))
    return contexts


def load_contexts(
    input_csv: str | Path,
    candidates_csv: str | Path,
    rules_json: str | Path,
    max_normal_samples: int = 8,
    *,
    min_avg_group: float = 3.0,
    min_dominance: float = 0.5,
    min_lift: float = 0.15,
) -> tuple[pd.DataFrame, list[RowContext]]:
    """一站式加载：返回 (原始表, RowContext 列表)。"""
    df = read_table(input_csv)
    candidates = read_table(candidates_csv)
    semantic_types = load_semantic_types(rules_json)
    normal_samples = compute_normal_samples(df, max_samples=max_normal_samples)
    column_stats = compute_column_stats(df, max_top=max_normal_samples)

    candidate_cols = [str(c) for c in candidates["column"].dropna().unique()]
    consensus_map = compute_consensus(
        df, candidate_cols,
        min_avg_group=min_avg_group, min_dominance=min_dominance, min_lift=min_lift,
    )

    contexts = build_row_contexts(
        df, candidates, semantic_types, normal_samples,
        consensus_map=consensus_map, column_stats=column_stats,
    )
    return df, contexts
