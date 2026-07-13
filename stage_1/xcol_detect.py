"""
跨列"全名 ↔ 标准缩写"一致性检测（针对两列被对调/错位的系统性错误）。

典型场景（rayyan）：journal_title 应为期刊全名、journal_abbreviation 应为其标准缩写，
但脏数据中部分行两列被对调（title 里放了缩写、abbreviation 里放了全名）。这类错误：
  - 不是缺失值、不违反单列格式/类型 -> 规则层与分布层都抓不到；
  - 两列各自取值都"合法"，只是位置错了 -> 必须用跨列关系才能发现。

方法（数据驱动发现 + LLM 语义确认，借鉴 Cocoon/FD 语义校验）：
  1. 用宽松统计发现候选列对 (full, abbrev)：full 显著更长、且多数行 abbrev 确为 full 的
     严格多 token 前缀缩写。
  2. 让 LLM 确认该列对是否真为"全名↔标准缩写"关系（过滤如"两列都是人名列表"之类伪对）。
  3. 对确认的列对，标记"关系反转"行（full 列反而像 abbrev 列的缩写）为对调错误(FI)，
     suggested_fix = 交换两列值。结果交 Stage 3 复核。
"""

from __future__ import annotations

import re

import pandas as pd

from stage_1.llm_rules import complete_json
from stage_1.profiling import is_blank

_TOKEN = re.compile(r"[A-Za-z]+")

XCOL_VALIDATE_PROMPT = """你是数据质量专家。下面是同一张表中两列在若干行上的取值对照。

## 列 A: {col_a}
## 列 B: {col_b}
## 样例（A 值 || B 值）:
{samples}

## 任务
判断 A、B 是否构成"同一实体的【全名】与其【标准缩写】"关系，即 B 通常是 A 的规范缩写
（例如：期刊全名 vs 期刊标准缩写、机构全称 vs 机构简称）。
注意区分：若两列只是"同类但不同内容"（如两列都是人名/地址/自由文本列表、或一列是无关编码），
则不构成该关系，应返回 false。

## 输出 JSON (只输出 JSON)
{{"is_abbreviation_pair": true, "full_column": "{col_a}", "reason": "简述判断依据"}}"""


def tokens(s: str) -> list[str]:
    return _TOKEN.findall(s)


def is_abbrev(short: str, full: str) -> bool:
    """
    严格判定 short 是否为 full 的多 token 缩写：
      - short 至少 2 个字母 token（排除单码如 'AMI'/'USA'）；
      - 每个 short token 按序对齐到 full 的某 token：len>=2 须为真前缀，len==1 须为首字母；
      - 全部 short token 都能对齐，且其中至少 2 个为"真前缀(len>=2)"强匹配。
    """
    st, ft = tokens(short.lower()), tokens(full.lower())
    if len(st) < 2 or len(st) > len(ft):
        return False
    j = 0
    strong = 0
    for tok in st:
        found = False
        while j < len(ft):
            full_tok = ft[j]
            j += 1
            if len(tok) >= 2 and full_tok.startswith(tok):
                strong += 1
                found = True
                break
            if len(tok) == 1 and full_tok[:1] == tok:
                found = True
                break
        if not found:
            return False
    return strong >= 2


def discover_abbrev_pairs(
    df: pd.DataFrame,
    *,
    min_rows: int = 30,
    min_len_ratio: float = 1.4,
    min_multi_token_rate: float = 0.5,
    min_abbrev_rate: float = 0.55,
) -> list[tuple[str, str, float]]:
    """发现候选 (full_col, abbrev_col) 列对（宽松统计，后续交 LLM 确认）。"""
    pairs: list[tuple[str, str, float]] = []
    cols = list(df.columns)
    for a in cols:
        for b in cols:
            if a == b:
                continue
            rows = [
                (str(df[a][i]), str(df[b][i]))
                for i in range(len(df))
                if not is_blank(df[a][i]) and not is_blank(df[b][i])
            ]
            if len(rows) < min_rows:
                continue
            med_a = pd.Series([len(x) for x, _ in rows]).median()
            med_b = pd.Series([len(y) for _, y in rows]).median()
            if med_b <= 0 or med_a < med_b * min_len_ratio:
                continue
            multi_b = sum(1 for _, y in rows if len(tokens(y)) >= 2) / len(rows)
            if multi_b < min_multi_token_rate:
                continue
            rate = sum(1 for x, y in rows if is_abbrev(y, x)) / len(rows)
            if rate >= min_abbrev_rate:
                pairs.append((a, b, round(rate, 3)))
    return pairs


def build_xcol_profile(full_col: str, abbrev_col: str, samples: list[tuple[str, str]]) -> dict:
    """构造缓存键 / prompt 画像（命名空间标记，避免与规则/标准化缓存冲突）。"""
    return {
        "_task": "xcol_abbrev",
        "full": str(full_col),
        "abbrev": str(abbrev_col),
        "samples": [[a, b] for a, b in samples],
    }


def validate_abbrev_pair(llm, full_col: str, abbrev_col: str,
                         samples: list[tuple[str, str]]) -> bool:
    """LLM 语义确认该列对是否真为"全名↔标准缩写"关系。解析失败保守返回 False。"""
    if not samples:
        return False
    sample_lines = "\n".join(f"  {a!r} || {b!r}" for a, b in samples)
    prompt = XCOL_VALIDATE_PROMPT.format(col_a=full_col, col_b=abbrev_col, samples=sample_lines)
    spec = complete_json(llm, prompt, label=f"列对 {full_col}↔{abbrev_col} 缩写关系校验")
    if isinstance(spec, dict):
        return bool(spec.get("is_abbreviation_pair"))
    return False


def detect_swaps(df: pd.DataFrame, full_col: str, abbrev_col: str) -> list[dict]:
    """
    在确认的 (full_col, abbrev_col) 列对上标记"对调"行。

    正常：full 列是全名、abbrev 列是缩写（is_abbrev(abbrev, full) 成立）。
    异常：反转——full 列里放了缩写、abbrev 列里放了全名
         （is_abbrev(full_val, abbrev_val) 成立而反向不成立）=> 两列疑似对调。
    """
    errors: list[dict] = []
    full_vals = df[full_col].to_numpy()
    abbrev_vals = df[abbrev_col].to_numpy()
    for i in range(len(df)):
        va, vb = full_vals[i], abbrev_vals[i]
        if is_blank(va) or is_blank(vb):
            continue
        va, vb = str(va), str(vb)
        if is_abbrev(va, vb) and not is_abbrev(vb, va):
            # full 列存的是缩写、abbrev 列存的是全名 -> 对调
            # 用 df.index 取真实 row_id（与评估/掩码对齐），而非位置下标
            row_id = df.index[i]
            errors.append({
                "row_id": row_id, "column": full_col, "value": va,
                "error_type": "FI", "violated_rule": "column_swap",
                "reason": f"疑似与列 '{abbrev_col}' 对调：'{full_col}' 应为全名却存了缩写",
                "suggested_fix": vb, "confidence": 0.85,
            })
            errors.append({
                "row_id": row_id, "column": abbrev_col, "value": vb,
                "error_type": "FI", "violated_rule": "column_swap",
                "reason": f"疑似与列 '{full_col}' 对调：'{abbrev_col}' 应为缩写却存了全名",
                "suggested_fix": va, "confidence": 0.85,
            })
    return errors
