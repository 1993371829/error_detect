"""
Stage 3 评估：以 clean vs dirty 的逐单元格差异为 ground truth，
对比"精检前(合并候选)"与"精检后(最终确认)"的 P/R/F1，并突出 Stateavg 误报的化解效果。

用法:
    python -m stage_3.evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage_2.io_utils import read_table
from stage_2.evaluate import ground_truth_cells, cells_from, metrics
from stage_3.config import Stage3Config


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
    cfg = Stage3Config()
    parser = argparse.ArgumentParser(description="Stage 3 evaluation")
    parser.add_argument("--clean", default=cfg.paths.clean_csv)
    parser.add_argument("--dirty", default=cfg.paths.input_csv)
    parser.add_argument("--candidates", default=cfg.paths.candidates)
    parser.add_argument("--final", default=cfg.paths.final_errors_out)
    args = parser.parse_args(argv)

    clean = read_table(args.clean)
    dirty = read_table(args.dirty)
    gt = ground_truth_cells(clean, dirty)
    print(f"Ground-truth 注入错误单元格: {len(gt)}\n")

    before = None
    if Path(args.candidates).exists():
        cand = read_table(args.candidates)
        before = _report("精检前(合并候选)", cells_from(cand), gt)

    after = None
    if Path(args.final).exists():
        final = read_table(args.final)
        after = _report("精检后(最终确认)", cells_from(final), gt)

    if before and after:
        print("\n变化:")
        print(f"  Precision {before['precision']:.3f} -> {after['precision']:.3f}")
        print(f"  Recall    {before['recall']:.3f} -> {after['recall']:.3f}")
        print(f"  F1        {before['f1']:.3f} -> {after['f1']:.3f}")

    # Stateavg 误报专项
    if Path(args.candidates).exists() and Path(args.final).exists():
        cand = read_table(args.candidates)
        final = read_table(args.final)
        cand_sa = {c for c in cells_from(cand) if c[1] == "Stateavg"} - gt
        final_sa = {c for c in cells_from(final) if c[1] == "Stateavg"} - gt
        print(f"\nStateavg 误报: 精检前 {len(cand_sa)} -> 精检后 {len(final_sa)} "
              f"(化解 {len(cand_sa) - len(final_sa)})")


if __name__ == "__main__":
    main()
