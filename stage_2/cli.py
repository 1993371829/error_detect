"""
Stage 2 命令行入口：分布异常检测（DAE 单元格级 + GANomaly 行级，默认融合）。

前置条件:
    pip install -r requirements.txt
    pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
    先完成 Stage 1，确保项目根目录存在 clean_mask.csv 与 data/hospital_errors.csv

运行命令（在项目根目录 d:\\study\\error_dect 下执行）:

    # hospital 全流程 — Stage 2（默认 fuse：DAE + GANomaly 融合）
    python -m stage_2.cli

    # 仅 DAE 单元格级（更快）
    python -m stage_2.cli --mode dae

    # 仅 GANomaly 行级
    python -m stage_2.cli --mode ganomaly

    # 调阈值：偏召回（误报交 Stage 3 过滤）
    python -m stage_2.cli --quantile 0.98 --max-cells-per-row 2

    # 调阈值：偏精度
    python -m stage_2.cli --quantile 0.995 --max-cells-per-row 1

    # 调试：缩短训练轮数
    python -m stage_2.cli --epochs 50

    # 显式指定输入/输出路径
    python -m stage_2.cli --input data/hospital_dirty.csv --clean-mask clean_mask.csv --stage1-errors data/hospital_errors.csv --candidates-out data/stage2_candidates.csv --combined-out data/combined_candidates.csv

评估（需 data/hospital_clean.csv）:
    python -m stage_2.evaluate
    python -m stage_2.evaluate --by-column --dae-candidates data/stage2_dae.csv --ganomaly-candidates data/stage2_ganomaly.csv

产出:
    data/stage2_candidates.csv   融合后的 DIST 候选
    data/stage2_dae.csv            DAE 单独候选
    data/stage2_ganomaly.csv       GANomaly 单独候选
    data/combined_candidates.csv   Stage1 ∪ Stage2 合并（Stage 3 输入）

下一步:
    python -m stage_3.cli

流程:
    读 dirty + clean_mask -> 编码(仅干净单元格估参) -> 干净行训练 ->
    全量打分(DAE 逐列 / GANomaly 行级) -> 融合 DIST 候选 -> 与 Stage1 合并去重。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stage_2.config import Stage2Config
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import build_model
from stage_2.score import run_stage2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: distribution anomaly detection")
    parser.add_argument("--mode", choices=["fuse", "dae", "ganomaly"], default=None,
                        help="检测模式：fuse=DAE+GANomaly 融合 / dae / ganomaly")
    parser.add_argument("--input", default=None, help="待检测脏表 CSV")
    parser.add_argument("--clean-mask", default=None, help="Stage 1 干净单元格掩码 CSV")
    parser.add_argument("--stage1-errors", default=None, help="Stage 1 错误结果 CSV")
    parser.add_argument("--candidates-out", default=None, help="Stage 2 融合 DIST 候选输出")
    parser.add_argument("--combined-out", default=None, help="合并候选输出（Stage3 输入）")
    parser.add_argument("--quantile", type=float, default=None, help="DAE 逐列阈值分位数")
    parser.add_argument("--max-cells-per-row", type=int, default=None,
                        help="DAE 每行最多保留 top-N 高贡献单元格（0=不限）")
    parser.add_argument("--row-quantile", type=float, default=None, help="GANomaly 行级阈值分位数")
    parser.add_argument("--row-top-k", type=int, default=None, help="GANomaly 可疑行内取 top-k 列")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（同时作用于两模型）")
    return parser


def _resolve_config(args: argparse.Namespace) -> Stage2Config:
    cfg = Stage2Config()
    if args.mode:
        cfg.mode = args.mode
    if args.input:
        cfg.paths.input_csv = args.input
    if args.clean_mask:
        cfg.paths.clean_mask = args.clean_mask
    if args.stage1_errors:
        cfg.paths.stage1_errors = args.stage1_errors
    if args.candidates_out:
        cfg.paths.candidates_out = args.candidates_out
    if args.combined_out:
        cfg.paths.combined_out = args.combined_out
    if args.quantile is not None:
        cfg.scoring.quantile = args.quantile
    if args.max_cells_per_row is not None:
        cfg.scoring.max_cells_per_row = args.max_cells_per_row
    if args.row_quantile is not None:
        cfg.scoring.row_quantile = args.row_quantile
    if args.row_top_k is not None:
        cfg.scoring.row_top_k = args.row_top_k
    if args.epochs is not None:
        cfg.dae.epochs = args.epochs
        cfg.ganomaly.epochs = args.epochs
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _resolve_config(args)

    df = read_table(cfg.paths.input_csv)
    print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

    mask_path = Path(cfg.paths.clean_mask)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"未找到干净掩码 {mask_path}，请先运行 Stage 1 生成 clean_mask.csv。"
        )
    clean_mask = read_clean_mask(mask_path)
    row_clean = int(clean_mask.all(axis=1).sum())
    print(f"干净掩码: {int(clean_mask.values.sum())}/{clean_mask.size} 单元格干净, "
          f"{row_clean} 行整行干净（用于训练）")

    dae = build_model("dae", **cfg.dae_kwargs()) if cfg.mode in ("dae", "fuse") else None
    ganomaly = build_model("ganomaly", **cfg.ganomaly_kwargs()) if cfg.mode in ("ganomaly", "fuse") else None
    print(f"模式: {cfg.mode} | 训练中...")

    candidates, parts = run_stage2(
        df, clean_mask,
        dae=dae, ganomaly=ganomaly, mode=cfg.mode,
        max_cardinality=cfg.encoding.max_cardinality,
        n_hash=cfg.encoding.n_hash,
        quantile=cfg.scoring.quantile,
        max_cells_per_row=cfg.scoring.max_cells_per_row,
        row_quantile=cfg.scoring.row_quantile,
        row_top_k=cfg.scoring.row_top_k,
        row_min_col_z=cfg.scoring.row_min_col_z,
    )

    out = Path(cfg.paths.candidates_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    print(f"\nStage 2 检出 {len(candidates)} 个 DIST 候选单元格 -> {out}")
    if len(candidates):
        if "subtype" in candidates.columns:
            print("通道分布:", dict(candidates["subtype"].value_counts()))
        print("列分布:")
        print(candidates["column"].value_counts().to_string())

    # 各模型单独候选（供三方对比与调试）
    if "dae" in parts and parts["dae"] is not None and not parts["dae"].empty:
        parts["dae"].to_csv(Path(cfg.paths.dae_out), index=False)
    if "ganomaly" in parts and parts["ganomaly"] is not None and not parts["ganomaly"].empty:
        parts["ganomaly"].to_csv(Path(cfg.paths.ganomaly_out), index=False)

    # 与 Stage 1 合并
    s1_path = Path(cfg.paths.stage1_errors)
    if s1_path.exists():
        stage1 = read_table(s1_path)
        if "row_id" in stage1.columns:
            stage1["row_id"] = stage1["row_id"].astype(int)
        combined = merge_candidates(stage1, candidates)
        comb_out = Path(cfg.paths.combined_out)
        comb_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(comb_out, index=False)
        print(f"\n合并 Stage1+Stage2 -> {len(combined)} 个候选 -> {comb_out}")
        if "source" in combined.columns:
            print(combined["source"].value_counts().to_string())
    else:
        print(f"[warn] 未找到 Stage 1 错误文件 {s1_path}，跳过合并。")


if __name__ == "__main__":
    main()
