"""
Stage 2 评估：以 clean vs dirty 的逐单元格差异为 ground truth，
报告 DIST 候选与 "Stage1 ∪ Stage2" 合并集的 precision / recall / F1。

增强:
    - 逐列召回明细：定位哪些列被召回 / 漏报，指导编码与模型/阈值调参。
    - 数值列 / 类别列分组的 P/R/F1。

用法:
    python -m stage_2.evaluate --dirty data/hospital_dirty.csv
    python -m stage_2.evaluate --dirty data/hospital_dirty.csv --by-column
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.config import Stage2Config
from stage_2.io_utils import read_table


def ground_truth_cells(clean: pd.DataFrame, dirty: pd.DataFrame) -> set:
    """clean 与 dirty 不同的单元格集合（统一空值处理），按位置 row_id。"""
    assert clean.shape == dirty.shape, "clean 与 dirty 形状需一致"

    def norm(v):
        return "" if is_blank(v) else str(v)

    gt = set()
    for col in clean.columns:
        c = clean[col].map(norm).to_numpy()
        d = dirty[col].map(norm).to_numpy()
        for i in range(len(c)):
            if c[i] != d[i]:
                gt.add((i, col))
    return gt


def cells_from(df: pd.DataFrame) -> set:
    """从候选/错误 DataFrame 取 (row_id, column) 集合。"""
    if df is None or df.empty:
        return set()
    return set(zip(df["row_id"].astype(int), df["column"].astype(str)))


def metrics(detected: set, gt: set) -> dict:
    tp = len(detected & gt)
    fp = len(detected - gt)
    fn = len(gt - detected)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"detected": len(detected), "tp": tp, "fp": fp,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def row_metrics(detected: set, gt: set) -> dict:
    """行级 P/R/F1：一行只要含被检出的错误单元格即视为“检出错误行”。"""
    det_rows = {r for r, _ in detected}
    gt_rows = {r for r, _ in gt}
    return metrics(det_rows, gt_rows)


def _report_rowlevel(name: str, cells: set, gt: set) -> None:
    m = row_metrics(cells, gt)
    print(f"{'  └ row-level':18s} 错误行={m['detected']:4d} TP={m['tp']:4d} FP={m['fp']:4d} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


def classify_columns(df: pd.DataFrame, numeric_min_ratio: float = 0.8) -> dict:
    """把列粗分为 numeric / categorical，用于分组评估（与 encoding 判定口径一致）。"""
    kinds: dict[str, str] = {}
    for col in df.columns:
        s = df[col].dropna().astype(str)
        s = s[s.str.strip() != ""]
        if len(s) == 0:
            kinds[col] = "categorical"
            continue
        nums = pd.to_numeric(s, errors="coerce").dropna()
        kinds[col] = "numeric" if len(nums) >= numeric_min_ratio * len(s) else "categorical"
    return kinds


def _report(name: str, cells: set, gt: set) -> None:
    m = metrics(cells, gt)
    print(f"{name:18s} 检出={m['detected']:4d}  TP={m['tp']:4d}  FP={m['fp']:4d}  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    fp_cells = cells - gt
    if fp_cells:
        dist = pd.Series([c[1] for c in fp_cells]).value_counts().head(8)
        print(f"{'':18s} 误报列 Top: {dict(dist)}")


def _report_grouped(name: str, cells: set, gt: set, kinds: dict) -> None:
    """按 numeric / categorical 列类型分组报告 P/R/F1。"""
    for group in ("categorical", "numeric"):
        g_gt = {c for c in gt if kinds.get(c[1]) == group}
        g_cells = {c for c in cells if kinds.get(c[1]) == group}
        if not g_gt and not g_cells:
            continue
        m = metrics(g_cells, g_gt)
        print(f"{'  └ ' + group:18s} GT={len(g_gt):4d} 检出={m['detected']:4d} "
              f"TP={m['tp']:4d} FP={m['fp']:4d} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


def per_column_report(name: str, cells: set, gt: set, kinds: dict) -> None:
    """逐列召回明细：每列 GT / TP / FP / recall，按 GT 数降序。"""
    cols = sorted({c[1] for c in gt} | {c[1] for c in cells})
    rows = []
    for col in cols:
        c_gt = {c for c in gt if c[1] == col}
        c_det = {c for c in cells if c[1] == col}
        tp = len(c_gt & c_det)
        fp = len(c_det - c_gt)
        rec = tp / len(c_gt) if c_gt else float("nan")
        rows.append((col, kinds.get(col, "?"), len(c_gt), tp, fp, rec))
    rows.sort(key=lambda r: (-r[2], -r[3]))
    print(f"\n[{name}] 逐列明细:")
    print(f"  {'column':16s} {'kind':12s} {'GT':>4s} {'TP':>4s} {'FP':>4s} {'recall':>7s}")
    for col, kind, n_gt, tp, fp, rec in rows:
        rec_s = "  -  " if n_gt == 0 else f"{rec:.3f}"
        print(f"  {col:16s} {kind:12s} {n_gt:4d} {tp:4d} {fp:4d} {rec_s:>7s}")


def main(argv: list[str] | None = None) -> None:
    cfg = Stage2Config()
    parser = argparse.ArgumentParser(description="Stage 2 evaluation")
    parser.add_argument("--dirty", default=None, help="脏表 CSV（指定后自动推导其余路径）")
    parser.add_argument("--dataset", default=None, help="显式指定数据集名")
    parser.add_argument("--clean", default=None, help="评估用 clean CSV")
    parser.add_argument("--candidates", default=None, help="Stage2 候选 CSV")
    parser.add_argument("--combined", default=None, help="合并候选 CSV")
    parser.add_argument("--stage1-errors", default=None, help="Stage1 错误 CSV")
    parser.add_argument("--by-column", action="store_true",
                        help="额外打印融合/DIST 候选的逐列召回明细")
    args = parser.parse_args(argv)

    if args.dirty:
        cfg.set_paths_from_dataset(args.dirty, dataset=args.dataset)

    clean_path = args.clean or cfg.paths.clean_csv
    dirty_path = args.dirty or cfg.paths.input_csv
    candidates_path = args.candidates or cfg.paths.candidates_out
    combined_path = args.combined or cfg.paths.combined_out
    stage1_path = args.stage1_errors or cfg.paths.stage1_errors

    clean = read_table(clean_path)
    dirty = read_table(dirty_path)
    gt = ground_truth_cells(clean, dirty)
    kinds = classify_columns(clean)
    n_cat = sum(1 for c in gt if kinds.get(c[1]) == "categorical")
    n_num = len(gt) - n_cat
    print(f"Ground-truth 注入错误单元格: {len(gt)} (类别列 {n_cat} / 数值列 {n_num})\n")

    if Path(stage1_path).exists():
        s1 = cells_from(read_table(stage1_path))
        _report("Stage1(规则层)", s1, gt)
        _report_grouped("Stage1", s1, gt, kinds)
        _report_rowlevel("Stage1", s1, gt)

    dist = None
    if Path(candidates_path).exists():
        dist = cells_from(read_table(candidates_path))
        _report("Stage2(DIST)", dist, gt)
        _report_grouped("Stage2(DIST)", dist, gt, kinds)
        _report_rowlevel("Stage2(DIST)", dist, gt)

    comb = None
    if Path(combined_path).exists():
        comb = cells_from(read_table(combined_path))
        _report("合并(S1+S2)", comb, gt)
        _report_grouped("合并(S1+S2)", comb, gt, kinds)
        _report_rowlevel("合并(S1+S2)", comb, gt)

    if args.by_column:
        if dist is not None:
            per_column_report("Stage2(DIST)", dist, gt, kinds)
        if comb is not None:
            per_column_report("合并(S1+S2)", comb, gt, kinds)


if __name__ == "__main__":
    main()
