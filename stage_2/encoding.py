"""
Stage 2 数据编码：表格 DataFrame -> 数值张量（numpy），并为每一列同时给出
"输入编码"与"预测目标(target head)规格"，服务于统一的条件预测模型
（见 stage_2/model.py 的 ConditionalPredictor）。

设计要点（identity-preserving，见 stage_2/DESIGN.md）:
    - 旧版把高基数列（如 flight=100、各时刻列=137~328）一律丢进 surrogate 哈希通道，
      导致键(flight)与取值(time)的"身份"被抹掉，模型无法学到 P(time | flight)。
    - 新版提高类别基数上限 max_cardinality（默认 500），让 flight 这类键、以及中等基数列
      （如时刻）都以 one-hot 身份进入输入，并作为可预测的"类别目标"。
    - 仅超高基数自由文本列（HospitalName/Address 等 > 上限）才走 surrogate（形态特征）通道。

每列产出两部分:
    1. 输入块（模型条件特征）：
        - numeric:     [value(归一化), is_null]                       width=2
        - categorical: [one-hot(n_cat) | __UNK__ | freq | is_null]    width=n_cat+3
        - highcard:    [6 形态数值 | is_null | in_clean | n_hash 哈希]  width=8+n_hash
    2. 目标头(ColumnSpec.target_kind):
        - categorical: softmax 预测 (n_cat+1) 类（含 __UNK__）-> 交叉熵 / NLL 打分
        - numeric:     回归归一化标量              -> MSE / 标准化残差打分
        - surrogate:   重构 6 个形态数值特征        -> MSE 打分（超高基数文本兜底）

模型推理时按列"屏蔽自身输入块"再预测，避免平凡复制；打分阶段把每格的
条件似然/残差与干净子集分位阈值比较，得到可疑单元格。
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# 视为空/缺失的哨兵
_BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a", "empty", "?", "-", "--"}

# 带单位的数值：整串必须是"数字 + 可选 %/单一单位词"，避免把 "1720 university blvd" 误判为数值
_UNIT_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*%?\s*[a-zA-Z]*\s*$")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

# surrogate 目标维度数（前 6 个形态数值特征：长度/数字比/字母比/标点比/空格比/频次）
_SURROGATE_TARGET_DIM = 6


def _extract_number(val: str) -> float:
    """从 '97%' / '33 patients' / '100' 提取数值；纯文本含数字（如地址）返回 NaN。"""
    if not _UNIT_NUM_RE.match(val):
        return float("nan")
    m = _NUM_RE.search(val)
    return float(m.group()) if m else float("nan")


def _parse_numeric_unit(non_null: pd.Series) -> pd.Series:
    """对一列非空值做单位感知数值解析，返回 float Series（无法解析为 NaN）。"""
    return non_null.map(_extract_number)


def _is_blank(value) -> bool:
    """统一空值判定：NaN / 空串 / 常见缺失哨兵。"""
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _BLANK_TOKENS


def _blank_mask(series: pd.Series) -> np.ndarray:
    return series.map(_is_blank).to_numpy()


@dataclass
class ColumnSpec:
    """单列在统一模型中的输入块位置与预测目标规格。"""

    name: str
    start: int                       # 输入块在特征矩阵中的起始列
    width: int                       # 输入块宽度（屏蔽该列时整块置零）
    target_kind: str                 # "categorical" | "numeric" | "surrogate" | "none"
    # --- categorical ---
    onehot_dim: int = 0              # = n_cat + 1（含 __UNK__）
    unk_index: int = -1              # __UNK__ 在 one-hot 段中的下标（= n_cat）
    classes: list = field(default_factory=list)   # 长度 n_cat（不含 UNK），下标即类别 id
    # --- numeric ---
    value_index: int = -1            # 归一化数值所在列
    isnull_index: int = -1           # is_null 指示位所在列（categorical/numeric/highcard 均有）
    median: float = 0.0
    iqr: float = 1.0
    # --- surrogate ---
    surrogate_dim: int = 0           # 目标维度数（形态特征数）


@dataclass
class ColumnEncoding:
    """单列的编码元信息，用于 transform 与构造 ColumnSpec。"""

    name: str
    kind: str                              # numeric|categorical|highcard|dropped
    start: int = 0
    width: int = 0
    # numeric 参数
    median: float = 0.0
    iqr: float = 1.0
    # categorical 参数
    categories: list = field(default_factory=list)    # 不含 __UNK__
    category_index: dict = field(default_factory=dict)
    freq_map: dict = field(default_factory=dict)
    # 高基数 surrogate 参数
    clean_values: set = field(default_factory=set)
    len_med: float = 0.0
    len_iqr: float = 1.0
    n_hash: int = 0


class TabularEncoder:
    """
    表格编码器：fit 学习每列编码方案，transform 生成特征矩阵，
    column_specs 暴露每列的输入块位置与预测目标规格。

    Args:
        max_cardinality: 类别列唯一值上限，<= 则 one-hot（身份保留），> 则走高基数 surrogate。
            默认提高到 500，使键列（flight）与中等基数列（时刻）保持身份。
        numeric_min_ratio: 列中可解析为数字的比例下限，达到则视为数值列。
        n_hash: 高基数 surrogate 的字符 n-gram 哈希桶数（仅作输入条件，不作预测目标）。
        id_min_len: ID 列启发式：整数中位串长 >= 此值且串长稳定则不当数值处理。
        target_max_card: 类别"目标头"的最大类数；> 则该列仅作输入条件、不作预测目标
            （target_kind=none），避免 softmax 类别爆炸。默认与 max_cardinality 同。
    """

    def __init__(
        self,
        max_cardinality: int = 500,
        numeric_min_ratio: float = 0.8,
        n_hash: int = 16,
        id_min_len: int = 4,
        target_max_card: Optional[int] = None,
    ):
        self.max_cardinality = max_cardinality
        self.numeric_min_ratio = numeric_min_ratio
        self.n_hash = n_hash
        self.id_min_len = id_min_len
        self.target_max_card = target_max_card if target_max_card is not None else max_cardinality
        self.encodings: dict[str, ColumnEncoding] = {}
        self.n_features: int = 0
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, clean_mask: Optional[pd.DataFrame] = None) -> "TabularEncoder":
        """学习编码方案（仅基于干净单元格估参）。"""
        cursor = 0
        for col in df.columns:
            series = df[col]
            if clean_mask is not None and col in clean_mask.columns:
                series = series[clean_mask[col].astype(bool)]
            enc = self._fit_column(str(col), series)
            enc.start = cursor
            cursor += enc.width
            self.encodings[str(col)] = enc
        self.n_features = cursor
        self._fitted = True
        return self

    def _fit_column(self, name: str, series: pd.Series) -> ColumnEncoding:
        """决定单列类型并估计编码参数。"""
        non_null = series[~_blank_mask(series)].astype(str)
        if len(non_null) == 0:
            return ColumnEncoding(name=name, kind="dropped", width=0)

        lengths = non_null.str.len()
        nunique = non_null.nunique()

        # ID 列启发式：纯整数 + 串长稳定且较长（ProviderNumber/ZipCode/PhoneNumber）。
        # 不当数值量纲，改走类别(若基数允许)或 surrogate。
        plain = pd.to_numeric(non_null, errors="coerce")
        is_id = bool(
            plain.notna().mean() >= self.numeric_min_ratio
            and (plain.dropna() % 1 == 0).all()
            and float(lengths.median()) >= self.id_min_len
            and float(lengths.std() or 0.0) <= 1.0
        )

        # 数值列（单位感知：97% / 33 patients 视为数值；含数字的地址不算）
        if not is_id:
            parsed = _parse_numeric_unit(non_null)
            if parsed.notna().mean() >= self.numeric_min_ratio:
                vals = parsed.dropna()
                median = float(vals.median())
                q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
                iqr = (q3 - q1) or (float(vals.std()) or 1.0)
                enc = ColumnEncoding(name=name, kind="numeric", median=median, iqr=iqr)
                enc.width = 2  # [value, is_null]
                return enc

        # 低/中基数（含 ID）类别列：one-hot + __UNK__ + freq + is_null（身份保留）
        if nunique <= self.max_cardinality:
            return self._fit_categorical(name, non_null)

        # 超高基数文本/ID 列：surrogate 形态特征
        return self._fit_highcard(name, non_null)

    def _fit_categorical(self, name: str, non_null: pd.Series) -> ColumnEncoding:
        categories = sorted(non_null.unique().tolist())
        index = {cat: i for i, cat in enumerate(categories)}
        n_cat = len(categories)
        enc = ColumnEncoding(
            name=name, kind="categorical",
            categories=categories, category_index=index,
            freq_map=self._freq_map(non_null),
        )
        # 布局: [one-hot(n_cat) | __UNK__(1) | freq(1) | is_null(1)]
        enc.width = n_cat + 3
        return enc

    def _fit_highcard(self, name: str, non_null: pd.Series) -> ColumnEncoding:
        lengths = non_null.str.len()
        enc = ColumnEncoding(
            name=name, kind="highcard",
            clean_values=set(non_null.tolist()),
            freq_map=self._freq_map(non_null),
            len_med=float(lengths.median()),
            len_iqr=float((lengths.quantile(0.75) - lengths.quantile(0.25)) or 1.0),
            n_hash=self.n_hash,
        )
        # 布局: 6 numeric(len/digit/alpha/punct/space/freq) + 2 binary(null/in_clean)
        #       + n_hash numeric(字符 3-gram 哈希分布)
        enc.width = 8 + (self.n_hash if self.n_hash > 0 else 0)
        return enc

    @staticmethod
    def _freq_map(non_null: pd.Series) -> dict:
        """value -> 归一化 log 频次（[0,1]，越小越罕见）。"""
        counts = non_null.value_counts()
        log_counts = np.log1p(counts.to_numpy(dtype=np.float64))
        denom = float(log_counts.max()) or 1.0
        return {val: float(lc / denom) for val, lc in zip(counts.index, log_counts)}

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """将表格编码为 (n_rows, n_features) 的 float 矩阵。"""
        if not self._fitted:
            raise RuntimeError("TabularEncoder 未 fit，先调用 fit()。")
        n = len(df)
        matrix = np.zeros((n, self.n_features), dtype=np.float32)
        for col, enc in self.encodings.items():
            if enc.width == 0:
                continue
            block = self._transform_column(df[col], enc)
            matrix[:, enc.start:enc.start + enc.width] = block
        return matrix

    def fit_transform(
        self, df: pd.DataFrame, clean_mask: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        return self.fit(df, clean_mask=clean_mask).transform(df)

    def _transform_column(self, series: pd.Series, enc: ColumnEncoding) -> np.ndarray:
        n = len(series)
        out = np.zeros((n, enc.width), dtype=np.float32)
        blank = _blank_mask(series)
        if enc.kind == "numeric":
            nums = series.astype(str).map(_extract_number).to_numpy(dtype=np.float64)
            filled = np.where(np.isnan(nums), enc.median, nums)
            out[:, 0] = (filled - enc.median) / enc.iqr
            out[:, 1] = blank.astype(np.float32)
        elif enc.kind == "categorical":
            self._fill_categorical(series, enc, out, blank)
        elif enc.kind == "highcard":
            self._fill_highcard(series, enc, out, blank)
        return out

    def _fill_categorical(self, series, enc, out, blank) -> None:
        n_cat = len(enc.categories)
        unk_pos = n_cat            # __UNK__ 在 one-hot 段末尾
        freq_pos = n_cat + 1
        null_pos = n_cat + 2
        vals = series.astype(str).tolist()
        for i, val in enumerate(vals):
            if blank[i]:
                out[i, null_pos] = 1.0
                out[i, unk_pos] = 1.0  # 缺失也视作非已知类别
                continue
            j = enc.category_index.get(val)
            if j is not None:
                out[i, j] = 1.0
            else:
                out[i, unk_pos] = 1.0
            out[i, freq_pos] = enc.freq_map.get(val, 0.0)

    def _fill_highcard(self, series, enc, out, blank) -> None:
        null_pos, in_clean_pos = 6, 7
        hash_start = 8
        vals = series.astype(str).tolist()
        for i, val in enumerate(vals):
            if blank[i]:
                out[i, null_pos] = 1.0
                continue
            length = len(val)
            digits = sum(c.isdigit() for c in val)
            alphas = sum(c.isalpha() for c in val)
            puncts = sum((not c.isalnum()) and (not c.isspace()) for c in val)
            spaces = val.count(" ")
            denom = length or 1
            out[i, 0] = (length - enc.len_med) / enc.len_iqr
            out[i, 1] = digits / denom
            out[i, 2] = alphas / denom
            out[i, 3] = puncts / denom
            out[i, 4] = spaces / denom
            out[i, 5] = enc.freq_map.get(val, 0.0)
            out[i, in_clean_pos] = 1.0 if val in enc.clean_values else 0.0
            if enc.n_hash > 0:
                out[i, hash_start:hash_start + enc.n_hash] = _hash_ngrams(val, enc.n_hash)

    # ------------------------------------------------------------- 列规格
    def feature_slice(self, column: str) -> tuple[int, int]:
        enc = self.encodings[column]
        return enc.start, enc.start + enc.width

    def column_specs(self) -> list[ColumnSpec]:
        """为每列构造输入块位置 + 预测目标规格（供 ConditionalPredictor 使用）。"""
        specs: list[ColumnSpec] = []
        for col, enc in self.encodings.items():
            if enc.width == 0:
                continue
            if enc.kind == "numeric":
                specs.append(ColumnSpec(
                    name=col, start=enc.start, width=enc.width,
                    target_kind="numeric",
                    value_index=enc.start, isnull_index=enc.start + 1,
                    median=enc.median, iqr=enc.iqr,
                ))
            elif enc.kind == "categorical":
                n_cat = len(enc.categories)
                onehot_dim = n_cat + 1
                # 类数超过目标上限则仅作输入、不作预测目标（避免 softmax 爆炸）
                target_kind = "categorical" if onehot_dim <= self.target_max_card + 1 else "none"
                specs.append(ColumnSpec(
                    name=col, start=enc.start, width=enc.width,
                    target_kind=target_kind,
                    onehot_dim=onehot_dim, unk_index=n_cat,
                    classes=list(enc.categories),
                    isnull_index=enc.start + n_cat + 2,
                ))
            elif enc.kind == "highcard":
                specs.append(ColumnSpec(
                    name=col, start=enc.start, width=enc.width,
                    target_kind="surrogate",
                    surrogate_dim=_SURROGATE_TARGET_DIM,
                    isnull_index=enc.start + 6,
                ))
        return specs

    def active_columns(self) -> list[str]:
        return [c for c, e in self.encodings.items() if e.width > 0]


def _hash_ngrams(value: str, n_hash: int, n: int = 3) -> np.ndarray:
    """字符 n-gram 哈希分布（确定性 crc32），返回归一化计数向量。"""
    vec = np.zeros(n_hash, dtype=np.float32)
    grams = [value[i:i + n] for i in range(len(value) - n + 1)] or [value]
    for g in grams:
        bucket = zlib.crc32(g.encode("utf-8")) % n_hash
        vec[bucket] += 1.0
    total = float(vec.sum()) or 1.0
    return vec / total
