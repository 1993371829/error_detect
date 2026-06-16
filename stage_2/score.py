"""
Stage 2 打分与输出：基于统一条件预测模型的逐格似然/残差，按无监督阈值
筛出可疑单元格，产出可并入 Stage 1 / 送往 Stage 3 的候选错误。

打分逻辑（见 stage_2/DESIGN.md）:
    - 类别列：score = -log P(观测值 | 其余列)。观测值得到的条件概率越低越可疑。
      额外"精度闸门 margin"：仅当模型更偏好的另一取值概率显著高于观测值时才报，
      该 argmax 取值即 suggested_fix。
    - 数值列：score = |预测 - 观测| 的标准化残差。
    - 超高基数文本：score = 形态特征重构 MSE（兜底）。
    每列阈值取"干净单元格"上分数分布的高分位（quantile），无监督、按列自适应。

召回杠杆（二者取并集，满足其一即召回）:
    - 相对：scores > 干净子集分位阈值（降低 quantile 更宽松）。
    - 绝对（仅类别列）：P(观测值) < abs_prob_floor，即 scores > -log(abs_prob_floor)，
      不受分位数限制，直接抓"在上下文里本就极不可能"的低概率冲突取值。

输出 schema:
    row_id, column, value, error_type(=DIST), anomaly_score, col_contribution,
    suggested_fix, subtype
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from stage_2.encoding import TabularEncoder, ColumnSpec
from stage_2.model import ConditionalPredictor

_EPS = 1e-9


def _clean_column_mask(clean_mask: pd.DataFrame, col: str, n: int) -> np.ndarray:
    """取某列的干净单元格布尔掩码（缺列则全 True）。"""
    if col in clean_mask.columns:
        return clean_mask[col].astype(bool).to_numpy()
    return np.ones(n, dtype=bool)


def _score_column(
    spec: ColumnSpec,
    x_all: np.ndarray,
    pred,
    df: pd.DataFrame,
) -> Optional[dict]:
    """
    计算单列逐行分数及其辅助量。

    Returns dict(scores, valid, blank, suggested_fix(list[str]), pred_differs(bool array))
    或 None（该列不可打分）。
    """
    n = x_all.shape[0]
    if spec.target_kind == "categorical":
        onehot = x_all[:, spec.start:spec.start + spec.onehot_dim]
        obs_idx = onehot.argmax(axis=1)
        probs = np.asarray(pred)
        rows = np.arange(n)
        p_obs = probs[rows, obs_idx]
        pred_idx = probs.argmax(axis=1)
        p_pred = probs[rows, pred_idx]
        scores = -np.log(p_obs + _EPS)
        margin = p_pred - p_obs
        blank = x_all[:, spec.isnull_index] >= 0.5
        valid = ~blank
        # suggested_fix：模型最偏好的类别（UNK 则留空）
        fixes = [
            (spec.classes[k] if k < spec.unk_index else "")
            for k in pred_idx
        ]
        return {
            "scores": scores, "valid": valid, "blank": blank,
            "suggested_fix": fixes, "margin": margin,
            "pred_differs": pred_idx != obs_idx,
        }

    if spec.target_kind == "numeric":
        obs = x_all[:, spec.value_index]
        pred_v = np.asarray(pred)
        blank = x_all[:, spec.isnull_index] >= 0.5
        valid = ~blank
        scores = np.abs(pred_v - obs)
        fixes = [f"{(v * spec.iqr + spec.median):.6g}" for v in pred_v]
        return {
            "scores": scores, "valid": valid, "blank": blank,
            "suggested_fix": fixes, "margin": None,
            "pred_differs": np.ones(n, dtype=bool),
        }

    if spec.target_kind == "surrogate":
        obs = x_all[:, spec.start:spec.start + spec.surrogate_dim]
        pred_v = np.asarray(pred)
        blank = x_all[:, spec.isnull_index] >= 0.5
        valid = ~blank
        scores = np.abs(pred_v - obs).mean(axis=1)
        return {
            "scores": scores, "valid": valid, "blank": blank,
            "suggested_fix": [""] * n, "margin": None,
            "pred_differs": np.ones(n, dtype=bool),
        }
    return None


def flag_suspicious_cells(
    df: pd.DataFrame,
    specs: list[ColumnSpec],
    x_all: np.ndarray,
    preds: dict,
    clean_mask: pd.DataFrame,
    *,
    quantile: float = 0.99,
    margin: float = 0.0,
    min_predictability: float = 0.5,
    abs_prob_floor: float = 0.0,
    min_clean: int = 20,
    max_cells_per_row: int = 0,
) -> pd.DataFrame:
    """对每列计算分数并按干净分位阈值 + 绝对概率地板（并集）+ 精度闸门筛出可疑单元格。"""
    n = len(df)
    records: list[dict] = []
    for spec in specs:
        if spec.target_kind == "none":
            continue
        info = _score_column(spec, x_all, preds.get(spec.name), df)
        if info is None:
            continue
        scores = info["scores"]
        valid = info["valid"]
        col_clean = _clean_column_mask(clean_mask, spec.name, n)

        # 阈值样本：干净且有效（非空）的单元格；不足则退化为全体有效格
        thr_mask = col_clean & valid
        if int(thr_mask.sum()) < min_clean:
            thr_mask = valid
        if int(thr_mask.sum()) == 0:
            continue

        # 可预测性闸门（通用、无监督）：类别列若在干净集上本就难以由其余列预测
        # （如 src / flight 这类标识列），则其条件概率不可靠，跳过以免刷误报。
        if spec.target_kind == "categorical":
            predictability = float((~info["pred_differs"])[thr_mask].mean())
            if predictability < min_predictability:
                continue

        thr = float(np.quantile(scores[thr_mask], quantile))

        # 相对分位 与 绝对概率地板 取并集；绝对地板仅类别列（有概率语义）。
        base = scores > thr
        if spec.target_kind == "categorical" and abs_prob_floor > 0:
            abs_thr = -np.log(abs_prob_floor + _EPS)
            base = base | (scores > abs_thr)

        flag = valid & base & info["pred_differs"]
        if info["margin"] is not None:
            flag = flag & (info["margin"] >= margin)

        for pos in np.where(flag)[0]:
            records.append({
                "row_id": df.index[pos],
                "column": spec.name,
                "value": df.iloc[pos][spec.name],
                "error_type": "DIST",
                "anomaly_score": float(scores[pos]),
                "col_contribution": float(scores[pos]),
                "suggested_fix": info["suggested_fix"][pos],
                "subtype": spec.target_kind,
            })

    result = pd.DataFrame(records, columns=[
        "row_id", "column", "value", "error_type",
        "anomaly_score", "col_contribution", "suggested_fix", "subtype",
    ])
    if max_cells_per_row and not result.empty:
        result = (
            result.sort_values("anomaly_score", ascending=False)
            .groupby("row_id", group_keys=False)
            .head(max_cells_per_row)
            .reset_index(drop=True)
        )
    return result


def run_stage2(
    df: pd.DataFrame,
    clean_mask: pd.DataFrame,
    *,
    model: Optional[ConditionalPredictor] = None,
    max_cardinality: int = 500,
    n_hash: int = 16,
    target_max_card: Optional[int] = None,
    quantile: float = 0.99,
    margin: float = 0.0,
    min_predictability: float = 0.5,
    abs_prob_floor: float = 0.0,
    max_cells_per_row: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Stage 2 端到端：编码 -> 仅干净行训练统一条件预测模型 -> 全量逐格打分 -> 候选。

    Returns:
        (candidates, {"model": candidates})
    """
    encoder = TabularEncoder(
        max_cardinality=max_cardinality, n_hash=n_hash, target_max_card=target_max_card
    )
    encoder.fit(df, clean_mask=clean_mask)
    specs = encoder.column_specs()

    row_clean = clean_mask.all(axis=1).to_numpy()
    x_all = encoder.transform(df)
    x_clean = x_all[row_clean]

    if model is None:
        model = ConditionalPredictor()
    model.fit(x_clean, specs)
    preds = model.predict(x_all)

    candidates = flag_suspicious_cells(
        df, specs, x_all, preds, clean_mask,
        quantile=quantile, margin=margin,
        min_predictability=min_predictability, abs_prob_floor=abs_prob_floor,
        max_cells_per_row=max_cells_per_row,
    )
    return candidates, {"model": candidates}
