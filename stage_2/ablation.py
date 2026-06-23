"""
检测器消融实验（文档 §17）。

一次性训练 reconstruction 并运行各检测器，再按检测器子集组合做证据融合，
对比 cell-level P/R/F1（融合 Stage1 后的合并集），量化各检测器的边际贡献。

用法:
    python -m stage_2.ablation --dirty data/hospital_dirty.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage_2.config import Stage2Config
from stage_2.coltypes import infer_column_kinds, load_semantic_types
from stage_2.detectors.base import DetectorContext
from stage_2.detectors.reconstruction import recon_frame_to_candidates
from stage_2.encoding import TabularEncoder
from stage_2.evaluate import cells_from, ground_truth_cells, metrics
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import ConditionalPredictor
from stage_2.schema import candidates_to_frame
from stage_2.score import _run_reconstruction

_OPTIONAL = ["statistical", "categorical", "association", "fd", "neighbor", "clustering"]


def collect_candidates(df, clean_mask, semantic_types, cfg: Stage2Config) -> dict[str, list]:
    """运行 reconstruction + 各可选检测器各一次，返回 detector_key -> CandidateError 列表。"""
    enc = TabularEncoder(max_cardinality=cfg.encoding.max_cardinality,
                         n_hash=cfg.encoding.n_hash,
                         target_max_card=cfg.encoding.target_max_card)
    enc.fit(df, clean_mask=clean_mask)
    specs = enc.column_specs()
    row_clean = clean_mask.all(axis=1).to_numpy()
    x_all = enc.transform(df)
    ctx = DetectorContext(kinds=infer_column_kinds(df, max_cardinality=cfg.encoding.max_cardinality),
                          semantic_types=semantic_types, encoder=enc, x_all=x_all, row_clean=row_clean)

    out: dict[str, list] = {}
    model = ConditionalPredictor()
    model.fit(x_all[row_clean], specs)
    recon = _run_reconstruction(
        df, specs, x_all, clean_mask, model,
        quantile=cfg.scoring.quantile, margin=cfg.scoring.margin,
        min_predictability=cfg.scoring.min_predictability,
        abs_prob_floor=cfg.scoring.abs_prob_floor, max_cells_per_row=cfg.scoring.max_cells_per_row,
        masked_inference=cfg.scoring.masked_inference,
        masked_inference_iters=cfg.scoring.masked_inference_iters,
        cdf_normalize=cfg.scoring.cdf_normalize,
    )
    out["reconstruction"] = recon_frame_to_candidates(recon)

    from stage_2.detectors.association_rule import detect_association
    from stage_2.detectors.categorical import detect_categorical
    from stage_2.detectors.clustering import detect_clustering
    from stage_2.detectors.fd_detector import detect_fd
    from stage_2.detectors.neighbor_consistency import detect_neighbor_consistency
    from stage_2.detectors.statistical import detect_statistical

    out["statistical"] = detect_statistical(df, clean_mask, ctx)
    out["categorical"] = detect_categorical(df, clean_mask, ctx)
    out["association"] = detect_association(df, clean_mask, ctx)
    out["fd"] = detect_fd(df, clean_mask, ctx)
    out["neighbor"] = detect_neighbor_consistency(df, clean_mask, ctx)
    out["clustering"] = detect_clustering(df, clean_mask, ctx)
    return out


def _eval_subset(keys, cand_map, stage1, gt) -> dict:
    pool = []
    for k in keys:
        pool += cand_map.get(k, [])
    s2 = candidates_to_frame(pool)
    combined = merge_candidates(stage1, s2)
    return metrics(cells_from(combined), gt)


def _print_row(label: str, m: dict) -> None:
    print(f"  {label:32s} 检出={m['detected']:4d} TP={m['tp']:4d} FP={m['fp']:4d} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2 检测器消融实验")
    parser.add_argument("--dirty", required=True, help="脏表 CSV")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)

    cfg = Stage2Config()
    dp = cfg.set_paths_from_dataset(args.dirty, dataset=args.dataset)
    df = read_table(cfg.paths.input_csv)
    clean_mask = read_clean_mask(cfg.paths.clean_mask)
    clean = read_table(cfg.paths.clean_csv)
    gt = ground_truth_cells(clean, df)
    semantic_types = load_semantic_types(dp.rules)
    stage1 = read_table(cfg.paths.stage1_errors) if Path(cfg.paths.stage1_errors).exists() else pd.DataFrame()
    if not stage1.empty:
        stage1["row_id"] = stage1["row_id"].astype(int)

    print(f"消融数据集: {dp.dataset}  GT={len(gt)}\n训练 reconstruction 并运行各检测器...")
    cand_map = collect_candidates(df, clean_mask, semantic_types, cfg)
    for k, v in cand_map.items():
        print(f"  [{k}] {len(v)} 候选")

    print("\n=== 合并集(Stage1 ∪ 子集) cell-level 指标 ===")
    _print_row("baseline: reconstruction", _eval_subset(["reconstruction"], cand_map, stage1, gt))
    for d in _OPTIONAL:
        _print_row(f"reconstruction + {d}", _eval_subset(["reconstruction", d], cand_map, stage1, gt))
    _print_row("all detectors", _eval_subset(["reconstruction"] + _OPTIONAL, cand_map, stage1, gt))
    print("\n  （留一法：从 all 中移除单个检测器）")
    full = ["reconstruction"] + _OPTIONAL
    for d in _OPTIONAL:
        subset = [k for k in full if k != d]
        _print_row(f"all - {d}", _eval_subset(subset, cand_map, stage1, gt))


if __name__ == "__main__":
    main()
