"""
Stage 2 clean_mask 污染量化诊断（只读，不调用 LLM，不改流水线）。

用 data/{dataset}_clean.csv 真值，量化 Stage 1 产出的
output/mask/{dataset}_clean_mask.csv 的两面污染与训练集稀缺度，验证
"clean mask 漏报脏值导致 Stage 2 分布基线被带偏"这一假设到底成不成立、主矛盾在哪：

  指标 A（面1, cell 级 mask 污染）：mask==True（被判干净）的格中实际为脏的比例。
      这正是"文档所称 15%"的直接度量；并附 mask 检测 P/R 参考。
  指标 B（面2, row-clean 训练集）：row_clean = mask.all(axis=1) 的行占比（数据稀缺度），
      以及这些 row-clean 行内实际仍为脏的格占比（= 预测器训练真实污染率）。
  指标 C（面1, 类别词表污染，判断中的主因）：用真实编码器口径（TabularEncoder.fit）
      取每个类别列的合法 one-hot 词表，数出"因漏报而混入词表的脏类别"（一个漏报 typo
      拿到独立合法类 -> 模型永远抓不到），并用编辑距离细分 typo 子类、给 Top 示例。

用法（项目根目录，需先跑完 Stage 1 产出 mask）:
    python -m stage_2.diagnose_mask --dirty data/hospital_dirty.csv
    python -m stage_2.diagnose_mask --dirty data/flights_dirty.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_1.profiling import is_blank
from stage_2.text_utils import levenshtein
from stage_2.config import Stage2Config
from stage_2.encoding import TabularEncoder, _is_blank
from stage_2.io_utils import read_clean_mask, read_table


def _safe(s) -> str:
    """ASCII-safe 字符串：非 ASCII 用 '?' 替换，规避 Windows 控制台 UnicodeEncodeError。"""
    return str(s).encode("ascii", "replace").decode("ascii")


def compute_gt_dirty(clean: pd.DataFrame, dirty: pd.DataFrame) -> pd.DataFrame:
    """逐格真值脏标记：clean 与 dirty 规整后不同即为脏（与 stage_2.evaluate 同口径）。"""
    assert clean.shape == dirty.shape, "clean 与 dirty 形状需一致"

    def norm(v):
        return "" if is_blank(v) else str(v)

    out = {}
    for col in clean.columns:
        c = clean[col].map(norm).to_numpy()
        d = dirty[col].map(norm).to_numpy()
        out[col] = c != d
    return pd.DataFrame(out, index=clean.index)


def _bool_col(mask: pd.DataFrame, col: str, n: int) -> np.ndarray:
    """取 mask 某列布尔数组；缺列视为全干净(True)，与 Stage 2 口径一致。"""
    if col in mask.columns:
        return mask[col].astype(bool).to_numpy()
    return np.ones(n, dtype=bool)


# --------------------------------------------------------------------- 指标 A
def metric_a(mask: pd.DataFrame, gt: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    面1 cell 级 mask 污染。

    Returns:
        overall: 总体统计（claimed_clean / leaked / leak_rate / mask 检测 P/R）
        per_col: 逐列 DataFrame（claimed_clean_cells, leaked_dirty_cells, leak_rate）
    """
    n = len(gt)
    rows = []
    tot_claimed_clean = tot_leaked = 0
    tot_claimed_dirty = tot_caught = tot_gt = 0
    for col in gt.columns:
        g = gt[col].to_numpy()
        m = _bool_col(mask, col, n)
        claimed_clean = int(m.sum())
        leaked = int((m & g).sum())             # 判干净却实际脏（漏报）
        claimed_dirty = int((~m).sum())
        caught = int((~m & g).sum())            # 判脏且实际脏（mask 命中）
        gt_n = int(g.sum())
        rows.append({
            "column": col,
            "claimed_clean_cells": claimed_clean,
            "leaked_dirty_cells": leaked,
            "leak_rate": round(leaked / claimed_clean, 4) if claimed_clean else 0.0,
            "gt_dirty": gt_n,
        })
        tot_claimed_clean += claimed_clean
        tot_leaked += leaked
        tot_claimed_dirty += claimed_dirty
        tot_caught += caught
        tot_gt += gt_n
    overall = {
        "claimed_clean_cells": tot_claimed_clean,
        "leaked_dirty_cells": tot_leaked,
        "leak_rate": round(tot_leaked / tot_claimed_clean, 4) if tot_claimed_clean else 0.0,
        "mask_detect_precision": round(tot_caught / tot_claimed_dirty, 4)
        if tot_claimed_dirty else 0.0,
        "mask_detect_recall": round(tot_caught / tot_gt, 4) if tot_gt else 0.0,
        "gt_dirty_total": tot_gt,
    }
    per_col = pd.DataFrame(rows).sort_values("leaked_dirty_cells", ascending=False)
    return overall, per_col


# --------------------------------------------------------------------- 指标 B
def metric_b(mask: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """面2 row-clean 训练集稀缺度与真实污染率。"""
    n = len(gt)
    # 与 Stage 2 一致：仅 mask 中存在的列参与 all() 判定
    cols = [c for c in gt.columns if c in mask.columns]
    if cols:
        m = mask[cols].astype(bool).to_numpy()
        row_clean = m.all(axis=1)
    else:
        row_clean = np.ones(n, dtype=bool)
    n_row_clean = int(row_clean.sum())

    g = gt[gt.columns].to_numpy()
    if n_row_clean:
        cells_in_clean_rows = int(g[row_clean].size)
        dirty_in_clean_rows = int(g[row_clean].sum())
    else:
        cells_in_clean_rows = dirty_in_clean_rows = 0
    return {
        "total_rows": n,
        "row_clean_rows": n_row_clean,
        "row_clean_ratio": round(n_row_clean / n, 4) if n else 0.0,
        "cells_in_clean_rows": cells_in_clean_rows,
        "dirty_in_clean_rows": dirty_in_clean_rows,
        "train_contamination_rate": round(dirty_in_clean_rows / cells_in_clean_rows, 4)
        if cells_in_clean_rows else 0.0,
    }


# --------------------------------------------------------------------- 指标 C
def metric_c(
    dirty: pd.DataFrame,
    mask: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    max_cardinality: int,
    numeric_min_ratio: float,
    n_hash: int,
    target_max_card,
    max_typo_distance: int = 2,
    top_examples: int = 10,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    面1 类别词表污染：用真实编码器口径取合法类，数出因漏报混入词表的脏类别。

    Returns:
        per_col: 逐类别列统计（vocab_size, spurious_classes, typo_classes）
        examples: Top 示例（column, dirty_class, nearest_clean, distance, occ）
        totals: 汇总
    """
    n = len(dirty)
    encoder = TabularEncoder(
        max_cardinality=max_cardinality,
        numeric_min_ratio=numeric_min_ratio,
        n_hash=n_hash,
        target_max_card=target_max_card,
    )
    encoder.fit(dirty, clean_mask=mask)

    rows = []
    examples: list[dict] = []
    tot_vocab = tot_spurious = tot_typo = 0
    for col, enc in encoder.encodings.items():
        if enc.kind != "categorical" or col not in gt.columns:
            continue
        vocab = set(enc.categories)
        if not vocab:
            continue

        # 还原"建词表用的格"：mask==True 且非空；与 encoding.fit 同口径
        vals = dirty[col].astype(str)
        blank = vals.map(_is_blank).to_numpy()
        m = _bool_col(mask, col, n)
        g = gt[col].to_numpy()
        sel = m & (~blank)
        sub = pd.DataFrame({"v": vals[sel].to_numpy(), "dirty": g[sel].astype(int)})
        agg = sub.groupby("v")["dirty"].agg(occ="size", dirty_occ="sum")

        # 纯漏报类：该类在"干净格"里的出现全部是脏（不漏报本不会进词表）
        clean_anchors = [v for v in vocab if int(agg.loc[v, "dirty_occ"]) == 0] \
            if len(agg) else []
        spurious = []
        for v in vocab:
            if v not in agg.index:
                continue
            occ = int(agg.loc[v, "occ"])
            dirty_occ = int(agg.loc[v, "dirty_occ"])
            if occ > 0 and dirty_occ == occ:
                spurious.append((v, occ))

        # typo 子类：纯漏报类与最近"干净合法类"编辑距离 <= 阈值
        typo_count = 0
        col_examples = []
        for v, occ in spurious:
            nearest, dist = None, None
            for a in clean_anchors:
                dd = levenshtein(v, a, max_distance=max_typo_distance)
                if dist is None or dd < dist:
                    nearest, dist = a, dd
            is_typo = dist is not None and 1 <= dist <= max_typo_distance
            if is_typo:
                typo_count += 1
            col_examples.append({
                "column": col,
                "dirty_class": v,
                "nearest_clean": nearest if is_typo else "",
                "distance": dist if is_typo else "",
                "occ": occ,
                "is_typo": is_typo,
            })

        rows.append({
            "column": col,
            "vocab_size": len(vocab),
            "spurious_classes": len(spurious),
            "typo_classes": typo_count,
        })
        examples.extend(col_examples)
        tot_vocab += len(vocab)
        tot_spurious += len(spurious)
        tot_typo += typo_count

    per_col = pd.DataFrame(rows).sort_values("spurious_classes", ascending=False) \
        if rows else pd.DataFrame(columns=["column", "vocab_size", "spurious_classes", "typo_classes"])
    # Top 示例优先 typo、再按 occ 降序
    examples.sort(key=lambda e: (not e["is_typo"], -e["occ"]))
    totals = {
        "categorical_columns": len(rows),
        "vocab_total": tot_vocab,
        "spurious_classes_total": tot_spurious,
        "typo_classes_total": tot_typo,
    }
    return per_col, examples[:top_examples], totals


# --------------------------------------------------------------------- 报告
def _print_overall(dataset: str, a: dict, b: dict, c_tot: dict) -> None:
    print("=" * 72)
    print(f"数据集: {_safe(dataset)}")
    print("-" * 72)
    print("[A] 面1 cell 级 mask 污染（验证文档 15%）")
    print(f"    被判干净格: {a['claimed_clean_cells']:6d}  其中实际脏(漏报): "
          f"{a['leaked_dirty_cells']:6d}  -> 漏报率 leak_rate = {a['leak_rate']:.2%}")
    print(f"    mask 检测参考: 召回 R={a['mask_detect_recall']:.3f}  "
          f"GT 脏格总数={a['gt_dirty_total']}")
    print("-" * 72)
    print("[B] 面2 row-clean 训练集（稀缺度 vs 训练真实污染率）")
    print(f"    整行干净行: {b['row_clean_rows']}/{b['total_rows']} "
          f"= {b['row_clean_ratio']:.2%}（用于训练的样本量）")
    print(f"    这些行内格数: {b['cells_in_clean_rows']:6d}  其中实际脏: "
          f"{b['dirty_in_clean_rows']:6d}  -> 训练真实污染率 = {b['train_contamination_rate']:.2%}")
    print("-" * 72)
    print("[C] 面1 类别词表污染（漏报脏值变合法 one-hot 类）")
    print(f"    类别列数: {c_tot['categorical_columns']}  合法类总数: {c_tot['vocab_total']}")
    print(f"    因漏报混入的脏类别: {c_tot['spurious_classes_total']}  "
          f"其中 typo 子类: {c_tot['typo_classes_total']}")
    print("=" * 72)


def _print_top_columns(per_col_a: pd.DataFrame, per_col_c: pd.DataFrame) -> None:
    head = per_col_a[per_col_a["leaked_dirty_cells"] > 0].head(10)
    if not head.empty:
        print("\n[A] 漏报最重的列 Top:")
        print(f"    {'column':18s} {'claimed_clean':>13s} {'leaked':>7s} {'leak_rate':>9s}")
        for _, r in head.iterrows():
            print(f"    {_safe(r['column']):18s} {r['claimed_clean_cells']:13d} "
                  f"{r['leaked_dirty_cells']:7d} {r['leak_rate']:8.2%}")
    if per_col_c is not None and not per_col_c.empty:
        head_c = per_col_c[per_col_c["spurious_classes"] > 0].head(10)
        if not head_c.empty:
            print("\n[C] 词表污染最重的列 Top:")
            print(f"    {'column':18s} {'vocab':>6s} {'spurious':>9s} {'typo':>5s}")
            for _, r in head_c.iterrows():
                print(f"    {_safe(r['column']):18s} {r['vocab_size']:6d} "
                      f"{r['spurious_classes']:9d} {r['typo_classes']:5d}")


def _print_examples(examples: list[dict]) -> None:
    if not examples:
        return
    print("\n[C] 漏报脏类别 Top 示例（typo 优先, '脏类别 -> 最近合法类'）:")
    for e in examples:
        if e["is_typo"]:
            print(f"    {_safe(e['column']):16s} {_safe(e['dirty_class'])!r:22s} -> "
                  f"{_safe(e['nearest_clean'])!r} (dist={e['distance']}, occ={e['occ']})")
        else:
            print(f"    {_safe(e['column']):16s} {_safe(e['dirty_class'])!r:22s} "
                  f"(非 typo, occ={e['occ']})")


def diagnose(dirty_path: str, dataset: str | None = None,
             clean_path: str | None = None, mask_path: str | None = None,
             out_dir: str | None = None) -> None:
    cfg = Stage2Config()
    dp = resolve_dataset_paths(dirty_path, cfg.layout, dataset=dataset)

    dirty = read_table(dp.dirty_csv)
    clean = read_table(Path(clean_path) if clean_path else dp.clean_csv)
    mp = Path(mask_path) if mask_path else dp.clean_mask
    if not mp.exists():
        raise FileNotFoundError(
            f"未找到 clean_mask {mp}，请先运行 Stage 1: python main.py --input {dirty_path}"
        )
    mask = read_clean_mask(mp)

    if clean.shape != dirty.shape:
        raise ValueError(f"clean {clean.shape} 与 dirty {dirty.shape} 形状不一致")

    gt = compute_gt_dirty(clean, dirty)

    a_overall, a_per_col = metric_a(mask, gt)
    b = metric_b(mask, gt)
    enc = cfg.encoding
    c_per_col, c_examples, c_totals = metric_c(
        dirty, mask, gt,
        max_cardinality=enc.max_cardinality,
        numeric_min_ratio=enc.numeric_min_ratio,
        n_hash=enc.n_hash,
        target_max_card=enc.target_max_card,
    )

    _print_overall(dp.dataset, a_overall, b, c_totals)
    _print_top_columns(a_per_col, c_per_col)
    _print_examples(c_examples)

    # 落盘逐列明细（A 与 C 合并）
    out_root = Path(out_dir) if out_dir else (Path("output") / "diagnostics")
    out_root.mkdir(parents=True, exist_ok=True)
    merged = a_per_col.merge(c_per_col, on="column", how="left")
    out_csv = out_root / f"{dp.dataset}_mask_contamination.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\n逐列明细已写入: {_safe(out_csv)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 clean_mask 污染量化诊断（只读）")
    p.add_argument("--dirty", required=True, help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--clean", default=None, help="评估用 clean CSV（默认自动推导）")
    p.add_argument("--clean-mask", default=None, help="Stage 1 干净掩码 CSV（默认自动推导）")
    p.add_argument("--out-dir", default=None, help="逐列明细输出目录（默认 output/diagnostics）")
    return p


def main(argv: list[str] | None = None) -> None:
    # 优先让中文标签在支持 UTF-8 的终端正常显示，不支持时降级替换而非崩溃
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    diagnose(
        args.dirty, dataset=args.dataset,
        clean_path=args.clean, mask_path=args.clean_mask, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
