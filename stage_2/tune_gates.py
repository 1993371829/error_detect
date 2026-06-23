"""
Stage 2 闸门调参网格（召回导向）。

目的:
    在不重训模型的前提下，扫描打分闸门 (quantile x abs_prob_floor x min_predictability)
    对召回/精度的影响，快速找到「召回明显提升、合并 F1 不降」的配置。

关键加速:
    编码与模型只与「训练数据（干净行）」有关，与打分闸门无关。因此每个数据集
    只 fit + predict 一次，之后对网格中每组闸门仅重跑 flag_suspicious_cells + 合并 + 评估，
    省去重复训练。

评估口径与 stage_2.evaluate 一致：clean vs dirty 逐格差异为 ground truth，
报告 Stage2(DIST) 与 合并(S1∪S2) 的 P/R/F1。

用法（项目根目录）:
    python -m stage_2.tune_gates --dirty data/flights_dirty.csv
    python -m stage_2.tune_gates --dirty data/hospital_dirty.csv \
        --quantiles 0.99,0.95,0.90 --floors 0.02,0.05,0.08 --preds 0.5,0.3
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_2.config import Stage2Config
from stage_2.encoding import TabularEncoder
from stage_2.evaluate import cells_from, ground_truth_cells, metrics
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import build_model
from stage_2.score import flag_suspicious_cells, neutralize_context


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main(argv: list[str] | None = None) -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    p = argparse.ArgumentParser(description="Stage 2 闸门调参网格（训练一次，扫描重打分）")
    p.add_argument("--dirty", required=True, help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--quantiles", default="0.99,0.95,0.90", help="逗号分隔的 quantile 列表")
    p.add_argument("--floors", default="0.02,0.05,0.08", help="逗号分隔的 abs_prob_floor 列表")
    p.add_argument("--preds", default="0.5,0.3", help="逗号分隔的 min_predictability 列表")
    p.add_argument("--margin", type=float, default=0.0, help="精度闸门 margin（固定）")
    p.add_argument("--masked-inference", action="store_true", help="预测时启用掩码推理")
    p.add_argument("--mi-iters", type=int, default=1, help="迭代式掩码推理轮数")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（加速试跑）")
    p.add_argument("--sort", default="merged_f1",
                   choices=["merged_f1", "merged_recall", "dist_recall", "dist_f1"],
                   help="结果排序键")
    args = p.parse_args(argv)

    cfg = Stage2Config()
    dp = resolve_dataset_paths(args.dirty, cfg.layout, dataset=args.dataset)

    df = read_table(dp.dirty_csv)
    clean = read_table(dp.clean_csv)
    clean_mask = read_clean_mask(dp.clean_mask)
    stage1 = read_table(dp.errors) if Path(dp.errors).exists() else pd.DataFrame()
    if not stage1.empty and "row_id" in stage1.columns:
        stage1["row_id"] = stage1["row_id"].astype(int)

    gt = ground_truth_cells(clean, df)
    s1_cells = cells_from(stage1) if not stage1.empty else set()

    # ---- 训练一次 ----
    encoder = TabularEncoder(
        max_cardinality=cfg.encoding.max_cardinality,
        n_hash=cfg.encoding.n_hash,
        target_max_card=cfg.encoding.target_max_card,
    )
    encoder.fit(df, clean_mask=clean_mask)
    specs = encoder.column_specs()
    row_clean = clean_mask.all(axis=1).to_numpy()
    x_all = encoder.transform(df)

    model_kwargs = cfg.model_kwargs()
    if args.epochs is not None:
        model_kwargs["epochs"] = args.epochs
    model = build_model(**model_kwargs)
    print(f"数据集 {dp.dataset}: 训练统一条件预测模型（一次）...")
    model.fit(x_all[row_clean], specs)

    # 掩码推理的预测（非迭代部分可复用一次预测；迭代需按候选重算）
    x_ctx0 = neutralize_context(x_all, specs, clean_mask) if args.masked_inference else x_all
    preds0 = model.predict(x_ctx0)

    def score_once(quantile, floor, pred, preds, cmask):
        return flag_suspicious_cells(
            df, specs, x_all, preds, cmask,
            quantile=quantile, margin=args.margin,
            min_predictability=pred, abs_prob_floor=floor,
        )

    grid = list(itertools.product(
        _parse_floats(args.quantiles), _parse_floats(args.floors), _parse_floats(args.preds)
    ))
    print(f"Ground-truth 错误: {len(gt)}  网格组合: {len(grid)}\n")

    rows = []
    for q, fl, pr in grid:
        # 迭代式掩码推理：按候选增广脏集重打分
        cand = score_once(q, fl, pr, preds0, clean_mask)
        if args.masked_inference and args.mi_iters > 1:
            prev = set(zip(cand.row_id, cand.column)) if not cand.empty else set()
            for _ in range(args.mi_iters - 1):
                eff = clean_mask.copy()
                for rid, c in zip(cand.row_id, cand.column):
                    if c in eff.columns and rid in eff.index:
                        eff.at[rid, c] = False
                preds_i = model.predict(neutralize_context(x_all, specs, eff))
                cand = score_once(q, fl, pr, preds_i, clean_mask)
                cur = set(zip(cand.row_id, cand.column)) if not cand.empty else set()
                if cur == prev:
                    break
                prev = cur

        dist = cells_from(cand)
        comb = s1_cells | dist
        md = metrics(dist, gt)
        mc = metrics(comb, gt)
        rows.append({
            "quantile": q, "floor": fl, "pred": pr,
            "dist_det": md["detected"], "dist_p": md["precision"],
            "dist_r": md["recall"], "dist_f1": md["f1"],
            "merged_det": mc["detected"], "merged_p": mc["precision"],
            "merged_r": mc["recall"], "merged_f1": mc["f1"],
        })

    res = pd.DataFrame(rows)
    sort_key = {"merged_f1": "merged_f1", "merged_recall": "merged_r",
                "dist_recall": "dist_r", "dist_f1": "dist_f1"}[args.sort]
    res = res.sort_values(sort_key, ascending=False).reset_index(drop=True)

    print(f"=== {dp.dataset} 闸门网格（按 {args.sort} 降序）  "
          f"masked_inference={args.masked_inference} iters={args.mi_iters} ===")
    print(f"{'q':>5} {'floor':>5} {'pred':>4} | "
          f"{'DIST det':>8} {'P':>5} {'R':>5} {'F1':>5} | "
          f"{'MRG det':>7} {'P':>5} {'R':>5} {'F1':>5}")
    for _, r in res.iterrows():
        print(f"{r['quantile']:5.2f} {r['floor']:5.2f} {r['pred']:4.1f} | "
              f"{int(r['dist_det']):8d} {r['dist_p']:.3f} {r['dist_r']:.3f} {r['dist_f1']:.3f} | "
              f"{int(r['merged_det']):7d} {r['merged_p']:.3f} {r['merged_r']:.3f} {r['merged_f1']:.3f}")


if __name__ == "__main__":
    main()
