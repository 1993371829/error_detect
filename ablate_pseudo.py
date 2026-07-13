"""
二阶段(Stage 2)重构模型训练集策略消融：strict(严格整行干净) vs pseudo(伪干净, 含单错屏蔽行)。

仅比较候选层(合并 S1+S2)的 P/R/F1，全程不调用 LLM，可复现（reconstruction seed 固定）。
用于论文中论证「统一采用严格整行干净训练」是否在 6 个数据集上都不劣于 pseudo。

用法（项目根目录）:
    python ablate_pseudo.py
    python ablate_pseudo.py --datasets movies,rayyan
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_2.coltypes import load_semantic_types
from stage_2.config import Stage2Config
from stage_2.evaluate import cells_from, ground_truth_cells, metrics
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import build_model
from stage_2.score import run_stage2

DATASETS = ["hospital", "flights", "beers", "rayyan", "billionaire", "movies"]

# 两种训练策略（只改重构训练集构造，其余检测器/阈值与当前默认一致）
VARIANTS = {
    "strict": {"reconstruction_pseudo_clean": False},
    "pseudo": {"reconstruction_pseudo_clean": True},
}


def _run_combined(dataset: str, variant_kwargs: dict) -> tuple[pd.DataFrame, int]:
    """按指定训练策略跑 Stage 2 并融合 Stage1，返回合并候选与严格干净行数。"""
    cfg = Stage2Config()
    dirty = f"data/{dataset}_dirty.csv"
    cfg.set_paths_from_dataset(dirty)
    for k, v in variant_kwargs.items():
        setattr(cfg.detectors, k, v)
    cfg.model.verbose = False  # 降噪

    df = read_table(cfg.paths.input_csv)
    clean_mask = read_clean_mask(Path(cfg.paths.clean_mask))
    n_strict = int(clean_mask.all(axis=1).sum())

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
        detectors=cfg.detectors,
        semantic_types=semantic_types,
    )

    s1_path = Path(cfg.paths.stage1_errors)
    stage1 = read_table(s1_path) if s1_path.exists() else pd.DataFrame()
    if not stage1.empty and "row_id" in stage1.columns:
        stage1["row_id"] = stage1["row_id"].astype(int)
    combined = merge_candidates(stage1, candidates)
    return combined, n_strict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage2 训练集策略消融 strict vs pseudo")
    parser.add_argument("--datasets", default=None, help="逗号分隔数据集名，默认全部")
    args = parser.parse_args(argv)
    datasets = ([s.strip() for s in args.datasets.split(",") if s.strip()]
                if args.datasets else DATASETS)

    rows = []
    for ds in datasets:
        clean = read_table(f"data/{ds}_clean.csv")
        dirty = read_table(f"data/{ds}_dirty.csv")
        gt = ground_truth_cells(clean, dirty)
        rec = {"dataset": ds, "gt": len(gt)}
        for vname, vkw in VARIANTS.items():
            print(f"\n########## {ds} / {vname} ##########")
            combined, n_strict = _run_combined(ds, vkw)
            rec["n_strict"] = n_strict
            rec[vname] = metrics(cells_from(combined), gt)
        rows.append(rec)

    # 汇总对照表
    print("\n" + "=" * 104)
    print("Stage2 训练集策略消融：合并候选(S1+S2) 候选层指标 —— strict(整行干净) vs pseudo(含单错屏蔽)")
    print("-" * 104)
    hdr = (f"{'dataset':12s}{'GT':>6s}{'strictRows':>11s}"
           f"{'  |  strict  P / R / F1':<26s}{'  |  pseudo  P / R / F1':<26s}{'  dF1(s-p)':>10s}")
    print(hdr)
    print("-" * 104)
    for r in rows:
        s, p = r["strict"], r["pseudo"]
        d_f1 = round(s["f1"] - p["f1"], 3)
        print(f"{r['dataset']:12s}{r['gt']:6d}{r['n_strict']:11d}"
              f"   {s['precision']:.3f}/{s['recall']:.3f}/{s['f1']:.3f}     "
              f"   {p['precision']:.3f}/{p['recall']:.3f}/{p['f1']:.3f}     "
              f"{d_f1:+8.3f}")
    print("=" * 104)

    out_dir = Path("output/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablate_pseudo_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"消融结果已写入 {out_path}")


if __name__ == "__main__":
    main()
