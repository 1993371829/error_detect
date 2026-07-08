"""
numeric_format 检测器消融：off vs on，候选层(合并 S1+S2) P/R/F1，全程免 LLM。

用法（项目根目录）:
    python ablate_numfmt.py                 # 全部 6 数据集
    python ablate_numfmt.py --datasets movies
    python ablate_numfmt.py --datasets movies --by-column   # movies 逐列明细
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_2.coltypes import load_semantic_types
from stage_2.config import Stage2Config
from stage_2.evaluate import (cells_from, classify_columns, ground_truth_cells,
                              metrics, per_column_report)
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import build_model
from stage_2.score import run_stage2

DATASETS = ["hospital", "flights", "beers", "rayyan", "billionaire", "movies"]

VARIANTS = {
    "off": {"numeric_format": False},
    "on": {"numeric_format": True},
}


def _run_combined(dataset: str, variant_kwargs: dict) -> pd.DataFrame:
    cfg = Stage2Config()
    dirty = f"data/{dataset}_dirty.csv"
    cfg.set_paths_from_dataset(dirty)
    for k, v in variant_kwargs.items():
        setattr(cfg.detectors, k, v)
    cfg.model.verbose = False

    df = read_table(cfg.paths.input_csv)
    clean_mask = read_clean_mask(Path(cfg.paths.clean_mask))
    model = build_model(**cfg.model_kwargs()) if cfg.detectors.reconstruction else None
    rules_path = resolve_dataset_paths(cfg.paths.input_csv).rules
    semantic_types = load_semantic_types(rules_path)

    candidates, _ = run_stage2(
        df, clean_mask,
        model=model,
        max_cardinality=cfg.encoding.max_cardinality,
        n_hash=cfg.encoding.n_hash,
        target_max_card=cfg.encoding.target_max_card,
        quantile=cfg.scoring.quantile,
        margin=cfg.scoring.margin,
        min_predictability=cfg.scoring.min_predictability,
        abs_prob_floor=cfg.scoring.abs_prob_floor,
        max_cells_per_row=cfg.scoring.max_cells_per_row,
        masked_inference=cfg.scoring.masked_inference,
        masked_inference_iters=cfg.scoring.masked_inference_iters,
        cdf_normalize=cfg.scoring.cdf_normalize,
        detectors=cfg.detectors,
        semantic_types=semantic_types,
        llm=None,
    )

    s1_path = Path(cfg.paths.stage1_errors)
    stage1 = read_table(s1_path) if s1_path.exists() else pd.DataFrame()
    if not stage1.empty and "row_id" in stage1.columns:
        stage1["row_id"] = stage1["row_id"].astype(int)
    return merge_candidates(stage1, candidates)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="numeric_format 检测器消融 off vs on")
    parser.add_argument("--datasets", default=None, help="逗号分隔数据集名，默认全部")
    parser.add_argument("--by-column", action="store_true", help="打印逐列明细(on)")
    args = parser.parse_args(argv)
    datasets = ([s.strip() for s in args.datasets.split(",") if s.strip()]
                if args.datasets else DATASETS)

    rows = []
    for ds in datasets:
        clean = read_table(f"data/{ds}_clean.csv")
        dirty = read_table(f"data/{ds}_dirty.csv")
        gt = ground_truth_cells(clean, dirty)
        rec = {"dataset": ds, "gt": len(gt)}
        combined_on = None
        for vname, vkw in VARIANTS.items():
            print(f"\n########## {ds} / numeric_format={vname} ##########")
            combined = _run_combined(ds, vkw)
            rec[vname] = metrics(cells_from(combined), gt)
            if vname == "on":
                combined_on = combined
        rows.append(rec)
        if args.by_column and combined_on is not None:
            kinds = classify_columns(clean)
            per_column_report(f"{ds} 合并(on)", cells_from(combined_on), gt, kinds)

    print("\n" + "=" * 96)
    print("numeric_format 消融：合并候选(S1+S2) 候选层指标 —— off vs on")
    print("-" * 96)
    print(f"{'dataset':12s}{'GT':>6s}"
          f"{'  |  off  P / R / F1':<24s}{'  |  on   P / R / F1':<24s}{'  dF1(on-off)':>13s}")
    print("-" * 96)
    for r in rows:
        o, n = r["off"], r["on"]
        d_f1 = round(n["f1"] - o["f1"], 3)
        print(f"{r['dataset']:12s}{r['gt']:6d}"
              f"   {o['precision']:.3f}/{o['recall']:.3f}/{o['f1']:.3f}     "
              f"   {n['precision']:.3f}/{n['recall']:.3f}/{n['f1']:.3f}     "
              f"{d_f1:+8.3f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
