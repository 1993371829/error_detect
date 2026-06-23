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
    row_id, column, value, error_type(=DIST), anomaly_score, norm_score,
    col_contribution, suggested_fix, subtype
    （norm_score：cdf_normalize 开启时为分数在干净分布上的累积分位 [0,1]，否则等于 anomaly_score）
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


def neutralize_context(
    x_all: np.ndarray,
    specs: list[ColumnSpec],
    clean_mask: pd.DataFrame,
) -> np.ndarray:
    """
    掩码推理：把 Stage 1 已确认脏（clean_mask==False）的单元格输入块中性化，
    避免「脏上下文滋生脏预测」。返回 x_all 的副本（不改原数组）。

    中性化策略（按列目标类型）:
        - categorical：整块置零后将 UNK 位置 1（表「未知类别」而非错误类别）。
        - numeric：value 置 0（= 归一化后的 median），is_null 置 0。
        - surrogate / 其余：整块置零。

    注意：仅用于「作为其余列的上下文」喂给模型预测；打分仍用原始 x_all 取观测值。
    """
    n = x_all.shape[0]
    x_ctx = x_all.copy()
    for spec in specs:
        dirty = ~_clean_column_mask(clean_mask, spec.name, n)
        if not dirty.any():
            continue
        block = slice(spec.start, spec.start + spec.width)
        x_ctx[dirty, block] = 0.0
        if spec.target_kind == "categorical" and spec.unk_index >= 0:
            x_ctx[dirty, spec.start + spec.unk_index] = 1.0
        # numeric/surrogate：整块置零即为中性（value=median 归一化后为 0，is_null=0）
    return x_ctx


def _candidate_cells(candidates: pd.DataFrame) -> set:
    """候选 DataFrame -> {(row_id, column)} 集合，用于迭代收敛判定。"""
    if candidates is None or candidates.empty:
        return set()
    return set(zip(candidates["row_id"].tolist(), candidates["column"].tolist()))


def _augment_mask(clean_mask: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """把当前 Stage 2 候选并入「脏集」：在 clean_mask 副本上将这些格置 False。"""
    eff = clean_mask.copy()
    if candidates is None or candidates.empty:
        return eff
    for row_id, col in zip(candidates["row_id"].tolist(), candidates["column"].tolist()):
        if col in eff.columns and row_id in eff.index:
            eff.at[row_id, col] = False
    return eff


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
    cdf_normalize: bool = False,
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

        # CDF 归一化：把分数映射到其在该列干净分布上的累积分位（[0,1]），
        # 使不同列（-logP / 残差 / MSE）的分数跨列可比；不改变判定逻辑。
        clean_sorted = np.sort(scores[thr_mask]) if cdf_normalize else None

        for pos in np.where(flag)[0]:
            s = float(scores[pos])
            if cdf_normalize:
                norm = float(np.searchsorted(clean_sorted, s) / len(clean_sorted))
            else:
                norm = s
            records.append({
                "row_id": df.index[pos],
                "column": spec.name,
                "value": df.iloc[pos][spec.name],
                "error_type": "DIST",
                "anomaly_score": s,
                "norm_score": norm,
                "col_contribution": s,
                "suggested_fix": info["suggested_fix"][pos],
                "subtype": spec.target_kind,
            })

    result = pd.DataFrame(records, columns=[
        "row_id", "column", "value", "error_type",
        "anomaly_score", "norm_score", "col_contribution", "suggested_fix", "subtype",
    ])
    if max_cells_per_row and not result.empty:
        # 跨列截断按归一化分数排序（cdf_normalize 关闭时 norm_score == anomaly_score）
        result = (
            result.sort_values("norm_score", ascending=False)
            .groupby("row_id", group_keys=False)
            .head(max_cells_per_row)
            .reset_index(drop=True)
        )
    return result


def _run_reconstruction(
    df, specs, x_all, clean_mask, model, *,
    quantile, margin, min_predictability, abs_prob_floor,
    max_cells_per_row, masked_inference, masked_inference_iters, cdf_normalize,
) -> pd.DataFrame:
    """自监督重构通道：训练条件预测模型并逐格打分（含可选迭代掩码推理）。"""
    def _score_with_mask(eff_mask: pd.DataFrame) -> pd.DataFrame:
        x_ctx = neutralize_context(x_all, specs, eff_mask) if masked_inference else x_all
        preds = model.predict(x_ctx)
        return flag_suspicious_cells(
            df, specs, x_all, preds, clean_mask,
            quantile=quantile, margin=margin,
            min_predictability=min_predictability, abs_prob_floor=abs_prob_floor,
            max_cells_per_row=max_cells_per_row, cdf_normalize=cdf_normalize,
        )

    candidates = _score_with_mask(clean_mask)
    if masked_inference and masked_inference_iters > 1:
        prev_cells = _candidate_cells(candidates)
        for _ in range(masked_inference_iters - 1):
            eff_mask = _augment_mask(clean_mask, candidates)
            candidates = _score_with_mask(eff_mask)
            cur_cells = _candidate_cells(candidates)
            if cur_cells == prev_cells:
                break
            prev_cells = cur_cells
    return candidates


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
    masked_inference: bool = False,
    masked_inference_iters: int = 1,
    cdf_normalize: bool = False,
    detectors=None,
    semantic_types: Optional[dict] = None,
    llm=None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Stage 2 端到端多检测器框架：编码 -> 按配置运行各检测器 -> 汇集统一候选。

    detectors: DetectorConfig（None 时默认仅 reconstruction，保持旧行为）。
    其余参数为 reconstruction 通道（条件预测模型）的编码/打分超参，含可选迭代掩码推理。

    Returns:
        (candidates, debug)  candidates 为统一 schema（含 detector 列）。
    """
    from stage_2.config import DetectorConfig
    from stage_2.coltypes import infer_column_kinds
    from stage_2.detectors.base import DetectorContext
    from stage_2.detectors.reconstruction import recon_frame_to_candidates
    from stage_2.schema import candidates_to_frame

    if detectors is None:
        detectors = DetectorConfig()

    encoder = TabularEncoder(
        max_cardinality=max_cardinality, n_hash=n_hash, target_max_card=target_max_card
    )
    encoder.fit(df, clean_mask=clean_mask)
    specs = encoder.column_specs()

    row_clean = clean_mask.all(axis=1).to_numpy()
    x_all = encoder.transform(df)

    ctx = DetectorContext(
        kinds=infer_column_kinds(df, max_cardinality=max_cardinality),
        semantic_types=semantic_types or {},
        encoder=encoder, x_all=x_all, row_clean=row_clean,
    )

    all_cands = []
    debug: dict[str, pd.DataFrame] = {}

    if detectors.reconstruction:
        if model is None:
            model = ConditionalPredictor()
        model.fit(x_all[row_clean], specs)
        recon_df = _run_reconstruction(
            df, specs, x_all, clean_mask, model,
            quantile=quantile, margin=margin, min_predictability=min_predictability,
            abs_prob_floor=abs_prob_floor, max_cells_per_row=max_cells_per_row,
            masked_inference=masked_inference, masked_inference_iters=masked_inference_iters,
            cdf_normalize=cdf_normalize,
        )
        recon_cands = recon_frame_to_candidates(recon_df)
        all_cands += recon_cands
        debug["reconstruction"] = recon_df
        print(f"  [reconstruction] {len(recon_cands)} 候选")

    if detectors.statistical:
        from stage_2.detectors.statistical import detect_statistical
        c = detect_statistical(df, clean_mask, ctx,
                               robust_z=detectors.robust_z, iqr_k=detectors.iqr_k)
        all_cands += c
        print(f"  [statistical] {len(c)} 候选")
    if detectors.categorical:
        from stage_2.detectors.categorical import detect_categorical
        c = detect_categorical(df, clean_mask, ctx, sim_threshold=detectors.sim_threshold)
        all_cands += c
        print(f"  [categorical] {len(c)} 候选")
    if detectors.association:
        from stage_2.detectors.association_rule import detect_association
        c = detect_association(df, clean_mask, ctx,
                               min_confidence=detectors.assoc_min_confidence)
        all_cands += c
        print(f"  [association] {len(c)} 候选")
    if detectors.fd:
        from stage_2.detectors.fd_detector import detect_fd
        c = detect_fd(df, clean_mask, ctx, semantic_check=llm is not None, llm=llm)
        all_cands += c
        print(f"  [approx_fd] {len(c)} 候选")
    if detectors.neighbor:
        from stage_2.detectors.neighbor_consistency import detect_neighbor_consistency
        c = detect_neighbor_consistency(df, clean_mask, ctx, k=detectors.knn_k)
        all_cands += c
        print(f"  [neighbor] {len(c)} 候选")
    if detectors.clustering:
        from stage_2.detectors.clustering import detect_clustering
        c = detect_clustering(df, clean_mask, ctx)
        all_cands += c
        print(f"  [clustering] {len(c)} 候选")

    candidates = candidates_to_frame(all_cands)
    return candidates, debug
