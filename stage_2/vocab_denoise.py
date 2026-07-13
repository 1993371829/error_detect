"""
LLM-free 词表去污：剔除混入 Stage 2 编码器类别词表的「漏报 typo」。

动机:
    clean_mask 由 Stage 1 errors 取反构建，Stage 1 漏掉的拼写错误（false negative）
    因 clean_mask=True 被当成合法值收进 TabularEncoder 的 one-hot 词表，自己独占一个
    合法类，模型从此对它给出「正常」的条件概率，永远抓不到（diagnose_mask 面 1 主因）。
    refine_mask（LLM 回灌）只能剔除 Stage 3 已确认的脏值；本模块用纯统计 + 编辑距离
    补上「Stage 1/3 都没抓到」的漏报 typo 这一缺口，零 LLM 成本。

方法:
    仅对「按 TabularEncoder 口径判为 categorical」的列处理（保持与真实词表一致）。
    在 clean_mask==True 的子集内：
        低频值 + 与唯一高频锚点编辑距离 <= max_edit + 锚点频次/候选频次 >= min_anchor_ratio
        => 判为漏报 typo，将这些单元格在掩码副本置 False（从训练集与词表同时剔除）。
    每列受 max_exclude_ratio cap 保护稀有合法值（沿用 refine_mask 的教训：
    过度剔除稀有值会拉低召回）。

用法（项目根目录）:
    python -m stage_2.vocab_denoise --dirty data/flights_dirty.csv
    python -m stage_2.vocab_denoise --dirty data/beers_dirty.csv --max-exclude-ratio 0.05
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from paths.layout import resolve_dataset_paths
from stage_1.profiling import is_blank
from stage_2.text_utils import levenshtein
from stage_2.config import Stage2Config
from stage_2.encoding import TabularEncoder
from stage_2.io_utils import read_clean_mask, read_table


def _categorical_columns(df: pd.DataFrame, clean_mask: pd.DataFrame, cfg) -> list[str]:
    """用真实编码器口径判定哪些列走 categorical 词表（与 Stage 2 训练一致）。"""
    encoder = TabularEncoder(
        max_cardinality=cfg.max_cardinality,
        numeric_min_ratio=cfg.numeric_min_ratio,
        n_hash=cfg.n_hash,
        target_max_card=cfg.target_max_card,
    )
    encoder.fit(df, clean_mask=clean_mask)
    return [c for c, e in encoder.encodings.items() if e.kind == "categorical"]


def _nearest_unique_anchor(
    value: str,
    anchors: list[str],
    counts: Counter,
    rare_count: int,
    *,
    max_edit: int,
    min_anchor_ratio: float,
) -> Optional[tuple[str, int]]:
    """
    为单个低频值找唯一近邻锚点（复用 typo_detect 的去误报思路）。

    若与多个锚点都相近（互相相似的枚举码邻域），返回 None；只有恰好 1 个近邻
    且锚点频次/候选频次达标时，才判为漏报 typo。
    """
    neighbors = []
    for anchor in anchors:
        if anchor == value:
            return None
        dist = levenshtein(value, anchor, max_distance=max_edit)
        if dist <= max_edit:
            neighbors.append((anchor, dist))
    if len(neighbors) != 1:
        return None
    anchor, dist = neighbors[0]
    if counts[anchor] / max(rare_count, 1) < min_anchor_ratio:
        return None
    return (anchor, dist)


def denoise_clean_mask(
    df: pd.DataFrame,
    clean_mask: pd.DataFrame,
    *,
    encoding_cfg=None,
    max_exclude_ratio: float = 0.05,
    min_anchor_count: int = 3,
    rare_max_count: int = 2,
    min_anchor_ratio: float = 10.0,
    max_edit: int = 1,
    min_anchor_len: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    返回去污后的掩码副本与统计信息。

    Args:
        df: 脏表（与 clean_mask 同形，RangeIndex）。
        clean_mask: round-0/refined 布尔掩码。
        encoding_cfg: EncodingConfig（决定词表口径）；None 时用默认。
        max_exclude_ratio: 每列新增剔除占总行数的上限（<0 不设限，0 不剔除）。
        min_anchor_count: 锚点（高频「正确」值）的最低频次。
        rare_max_count: 候选（低频疑似 typo）的最高频次。
        min_anchor_ratio: 锚点频次 / 候选频次的下限。
        max_edit: 与锚点的最大编辑距离。
        min_anchor_len: 锚点最短长度（过滤极短噪声）。

    Returns:
        (refined_mask, stats)
    """
    from stage_2.config import EncodingConfig

    cfg = encoding_cfg or EncodingConfig()
    refined = clean_mask.copy()
    n_rows = len(refined)
    cap = None if max_exclude_ratio < 0 else max(1, int(max_exclude_ratio * n_rows))

    cat_cols = _categorical_columns(df, clean_mask, cfg)

    per_col_excluded: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    n_excluded = 0
    n_capped = 0

    for col in cat_cols:
        if col not in refined.columns:
            continue
        col_clean = refined[col].astype(bool).to_numpy()
        series = df[col]
        # 仅在干净子集内统计词表与频次（与编码器口径一致）
        clean_vals = [
            str(v) for v, ok in zip(series.tolist(), col_clean)
            if ok and not is_blank(v)
        ]
        if not clean_vals:
            continue
        counts = Counter(clean_vals)
        anchors = [
            v for v, c in counts.items()
            if c >= min_anchor_count and len(v) >= min_anchor_len
        ]
        if not anchors:
            continue
        rares = {v for v, c in counts.items() if c <= rare_max_count}
        if not rares:
            continue

        # 候选值 -> (锚点, 距离)；命中即视为漏报 typo
        contaminant: dict[str, tuple[str, int]] = {}
        for value in rares:
            match = _nearest_unique_anchor(
                value, anchors, counts, counts[value],
                max_edit=max_edit, min_anchor_ratio=min_anchor_ratio,
            )
            if match is not None:
                contaminant[value] = match
        if not contaminant:
            continue

        # 收集要剔除的单元格（仅干净格中命中污染值的位置）
        cells: list[tuple[int, str, float]] = []  # (row_id, value, anchor_ratio)
        for pos, (v, ok) in enumerate(zip(series.tolist(), col_clean)):
            if not ok or is_blank(v):
                continue
            v = str(v)
            if v in contaminant:
                anchor, _ = contaminant[v]
                ratio = counts[anchor] / max(counts[v], 1)
                cells.append((int(refined.index[pos]), v, ratio))
        if not cells:
            continue

        # 按锚点频次比降序优先剔除「最像 typo」者，受 cap 限制保护稀有值
        cells.sort(key=lambda x: x[2], reverse=True)
        chosen = cells if cap is None else cells[:cap]
        n_capped += len(cells) - len(chosen)
        for row_id, _, _ in chosen:
            refined.at[row_id, col] = False
        per_col_excluded[col] = len(chosen)
        n_excluded += len(chosen)
        examples[col] = [
            f"'{v}'->'{contaminant[v][0]}'" for _, v, _ in chosen[:3]
        ]

    before_clean = int(clean_mask.to_numpy().sum())
    after_clean = int(refined.to_numpy().sum())
    stats = {
        "total_cells": refined.size,
        "categorical_cols": len(cat_cols),
        "max_edit": max_edit,
        "cap_per_col": cap,
        "excluded": n_excluded,
        "excluded_capped": n_capped,
        "clean_before": before_clean,
        "clean_after": after_clean,
        "clean_delta": after_clean - before_clean,
        "per_col_excluded": per_col_excluded,
        "examples": examples,
    }
    return refined, stats


def _print_stats(dataset: str, in_mask: Path, out_mask: Path, stats: dict) -> None:
    print("=" * 64)
    print(f"数据集: {dataset}  LLM-free 词表去污")
    print(f"  输入掩码: {in_mask}")
    print(f"  输出掩码: {out_mask}")
    print("-" * 64)
    print(f"  类别列数: {stats['categorical_cols']}  "
          f"(max_edit={stats['max_edit']}, 每列剔除上限={stats['cap_per_col']})")
    print(f"  剔除漏报 typo (True->False): {stats['excluded']}  "
          f"(因上限放弃 {stats['excluded_capped']})")
    print(f"  干净格: {stats['clean_before']} -> {stats['clean_after']} "
          f"(净变化 {stats['clean_delta']:+d}) / 共 {stats['total_cells']}")
    if stats["per_col_excluded"]:
        top = sorted(stats["per_col_excluded"].items(),
                     key=lambda kv: kv[1], reverse=True)[:10]
        print("  剔除最多的列 Top:")
        for col, n in top:
            ex = ", ".join(stats["examples"].get(col, []))
            print(f"    {col:24s} {n:4d}  例: {ex}")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-free 词表去污：剔除混入类别词表的漏报 typo（生成 _clean_mask_vd.csv）"
    )
    p.add_argument("--dirty", required=True, help="脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--clean-mask", default=None, help="输入掩码（默认自动推导 round-0）")
    p.add_argument("--out", default=None, help="输出掩码路径（默认 {dataset}_clean_mask_vd.csv）")
    p.add_argument("--max-exclude-ratio", type=float, default=0.05,
                   help="每列新增剔除占总行数上限（默认 0.05；<0 不设限）")
    p.add_argument("--min-anchor-ratio", type=float, default=10.0,
                   help="锚点频次/候选频次下限（默认 10）")
    p.add_argument("--max-edit", type=int, default=1, help="与锚点的最大编辑距离（默认 1）")
    return p


def main(argv: list[str] | None = None) -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    cfg = Stage2Config()
    dp = resolve_dataset_paths(args.dirty, cfg.layout, dataset=args.dataset)

    in_mask = Path(args.clean_mask) if args.clean_mask else dp.clean_mask
    out_mask = (Path(args.out) if args.out
                else dp.clean_mask.parent / f"{dp.dataset}_clean_mask_vd.csv")

    if not in_mask.exists():
        raise FileNotFoundError(f"未找到输入掩码 {in_mask}，请先运行 Stage 1。")

    df = read_table(dp.dirty_csv)
    mask_df = read_clean_mask(in_mask)

    refined, stats = denoise_clean_mask(
        df, mask_df,
        encoding_cfg=cfg.encoding,
        max_exclude_ratio=args.max_exclude_ratio,
        min_anchor_ratio=args.min_anchor_ratio,
        max_edit=args.max_edit,
    )

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(out_mask, index=False)
    _print_stats(dp.dataset, in_mask, out_mask, stats)


if __name__ == "__main__":
    main()
