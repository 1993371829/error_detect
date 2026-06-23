"""
Stage 3 评估：以 clean vs dirty 的逐单元格差异为 ground truth，
对比"精检前(合并候选)"与"精检后(最终确认)"的 P/R/F1，
并按列统计 Stage 3 化解的误报 Top-N（适用于任意数据集，无写死列名）。

用法:
    python -m stage_3.evaluate --dirty data/<dataset>_dirty.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from stage_1.profiling import is_blank
from stage_2.io_utils import read_table
from stage_2.evaluate import ground_truth_cells, cells_from, metrics
from stage_3.config import Stage3Config


def _norm(v) -> str:
    return "" if is_blank(v) else str(v)


def correction_report(final: pd.DataFrame, clean: pd.DataFrame, gt: set) -> None:
    """修复准确率：对确认为错且属于 GT 的格，比较 suggested_fix 与 clean 真值。"""
    if final is None or final.empty or "suggested_fix" not in final.columns:
        return
    total_fix = correct = tp_with_fix = 0
    for _, r in final.iterrows():
        try:
            row_id = int(r["row_id"])
        except (TypeError, ValueError):
            continue
        col = str(r["column"])
        if (row_id, col) not in gt:
            continue
        fix = r.get("suggested_fix")
        if fix in (None, "") or (isinstance(fix, float) and pd.isna(fix)):
            continue
        if row_id >= len(clean) or col not in clean.columns:
            continue
        tp_with_fix += 1
        total_fix += 1
        if _norm(fix) == _norm(clean.iloc[row_id][col]):
            correct += 1
    if total_fix:
        print(f"\n修复准确率(correction-level): {correct}/{total_fix} = "
              f"{correct / total_fix:.3f}（仅统计 TP 且给出修复值的格）")


def by_error_type_report(final: pd.DataFrame, gt: set) -> None:
    """按 error_type 分项：每类确认数 / TP / FP / 精度。"""
    if final is None or final.empty or "error_type" not in final.columns:
        return
    print("\n按 error_type 分项（确认错误）:")
    print(f"  {'type':8s} {'确认':>5s} {'TP':>5s} {'FP':>5s} {'precision':>9s}")
    cells_type: dict = {}
    for _, r in final.iterrows():
        try:
            cell = (int(r["row_id"]), str(r["column"]))
        except (TypeError, ValueError):
            continue
        cells_type.setdefault(str(r.get("error_type", "") or "?"), []).append(cell)
    for etype, cells in sorted(cells_type.items(), key=lambda kv: -len(kv[1])):
        tp = sum(1 for c in cells if c in gt)
        fp = len(cells) - tp
        prec = tp / len(cells) if cells else 0.0
        print(f"  {etype:8s} {len(cells):5d} {tp:5d} {fp:5d} {prec:9.3f}")


def _report(name: str, cells: set, gt: set) -> dict:
    m = metrics(cells, gt)
    print(f"{name:20s} 检出={m['detected']:4d}  TP={m['tp']:4d}  FP={m['fp']:4d}  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    fp_cells = cells - gt
    if fp_cells:
        dist = pd.Series([c[1] for c in fp_cells]).value_counts().head(6)
        print(f"{'':20s} 误报列 Top: {dict(dist)}")
    return m


def main(argv: list[str] | None = None) -> None:
    cfg = Stage3Config.resolve()
    parser = argparse.ArgumentParser(description="Stage 3 evaluation")
    parser.add_argument("--dirty", default=None, help="脏表 CSV（指定后自动推导其余路径）")
    parser.add_argument("--dataset", default=None, help="显式指定数据集名")
    parser.add_argument("--clean", default=None, help="评估用 clean CSV")
    parser.add_argument("--candidates", default=None, help="合并候选 CSV")
    parser.add_argument("--final", default=None, help="最终确认错误 CSV")
    args = parser.parse_args(argv)

    if args.dirty:
        cfg.set_paths_from_dataset(args.dirty, dataset=args.dataset)

    clean_path = args.clean or cfg.paths.clean_csv
    dirty_path = args.dirty or cfg.paths.input_csv
    candidates_path = args.candidates or cfg.paths.candidates
    final_path = args.final or cfg.paths.final_errors_out

    clean = read_table(clean_path)
    dirty = read_table(dirty_path)
    gt = ground_truth_cells(clean, dirty)
    print(f"Ground-truth 注入错误单元格: {len(gt)}\n")

    before = None
    if Path(candidates_path).exists():
        cand = read_table(candidates_path)
        before = _report("精检前(合并候选)", cells_from(cand), gt)

    after = None
    final = None
    if Path(final_path).exists():
        final = read_table(final_path)
        after = _report("精检后(最终确认)", cells_from(final), gt)
        correction_report(final, clean, gt)
        by_error_type_report(final, gt)

    if before and after:
        print("\n变化:")
        print(f"  Precision {before['precision']:.3f} -> {after['precision']:.3f}")
        print(f"  Recall    {before['recall']:.3f} -> {after['recall']:.3f}")
        print(f"  F1        {before['f1']:.3f} -> {after['f1']:.3f}")

    # 通用：Stage 3 在各列上化解的误报（精检前 FP -> 精检后 FP），按化解数量取 Top-N
    if before is not None and after is not None:
        cand_fp = cells_from(cand) - gt        # 精检前误报单元格
        final_fp = cells_from(final) - gt      # 精检后仍存在的误报
        pre = Counter(c[1] for c in cand_fp)
        post = Counter(c[1] for c in final_fp)
        resolved = {col: pre[col] - post.get(col, 0) for col in pre}
        top = sorted(resolved.items(), key=lambda kv: kv[1], reverse=True)[:6]
        if top:
            print("\n按列误报化解 Top（精检前FP -> 精检后FP, 化解）:")
            for col, gain in top:
                print(f"  {col:18s} {pre[col]:4d} -> {post.get(col, 0):4d}  (化解 {gain})")


if __name__ == "__main__":
    main()
