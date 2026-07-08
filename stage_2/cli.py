"""
Stage 2 命令行入口：统一条件预测模型的分布/冲突异常检测。

前置条件:
    pip install -r requirements.txt
    # CPU:
    pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
    # GPU 服务器（按实际 CUDA 版本选择 cuXXX，如 cu121）:
    pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
    先完成 Stage 1（产出 output/mask/{dataset}_clean_mask.csv 等）

运行命令（PowerShell，项目根目录）:

    python -m stage_2.cli --input data/hospital_dirty.csv
    python -m stage_2.cli --input data/flights_dirty.csv

    # GPU/CPU 选择（默认 auto：有 GPU 自动用，无则 CPU）
    python -m stage_2.cli --input data/movies_dirty.csv --device cuda

    # 调阈值
    python -m stage_2.cli --input data/hospital_dirty.csv --quantile 0.9 --abs-prob-floor 0.1

产出（以 hospital 为例）:
    output/stage2/hospital_stage2_candidates.csv
    output/stage2/hospital_combined_candidates.csv

下一步:
    python -m stage_3.cli --input data/hospital_dirty.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths.layout import ensure_output_dirs
from stage_2.config import Stage2Config
from stage_2.io_utils import merge_candidates, read_clean_mask, read_table
from stage_2.model import build_model
from stage_2.score import run_stage2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: conditional-prediction anomaly detection")
    parser.add_argument("--input", default=None, help="待检测脏表 CSV（data/{dataset}_dirty.csv）")
    parser.add_argument("--dataset", default=None, help="显式指定数据集名")
    parser.add_argument("--clean-mask", default=None, help="Stage 1 干净单元格掩码 CSV")
    parser.add_argument("--stage1-errors", default=None, help="Stage 1 错误结果 CSV")
    parser.add_argument("--candidates-out", default=None, help="Stage 2 DIST 候选输出")
    parser.add_argument("--combined-out", default=None, help="合并候选输出（Stage3 输入）")
    parser.add_argument("--quantile", type=float, default=None, help="逐列干净分位阈值")
    parser.add_argument("--margin", type=float, default=None,
                        help="类别精度闸门：备选类概率需超观测类至少该值才报")
    parser.add_argument("--min-predictability", type=float, default=None,
                        help="类别列在干净集上的最低 top-1 可预测性，低于则跳过该列")
    parser.add_argument("--abs-prob-floor", type=float, default=None,
                        help="类别列绝对概率地板：P(观测值)<该值即召回（与分位阈值取并集，0=关闭）")
    parser.add_argument("--max-cells-per-row", type=int, default=None,
                        help="每行最多保留 top-N 高分单元格（0=不限）")
    parser.add_argument("--masked-inference", action="store_true",
                        help="掩码推理：预测时中性化 Stage1 确认脏的上下文单元格（隔离脏上下文传播）")
    parser.add_argument("--masked-inference-iters", type=int, default=None,
                        help="迭代式掩码推理轮数（>1 把上一轮 Stage2 候选并入脏集重打分，松绑同行互相掩护）")
    parser.add_argument("--cdf-normalize", action="store_true",
                        help="输出 norm_score 为分数在干净分布上的累积分位（跨列可比）")
    parser.add_argument("--vocab-denoise", action="store_true",
                        help="训练前用编辑距离自过滤剔除混入类别词表的漏报 typo")
    parser.add_argument("--max-cardinality", type=int, default=None,
                        help="类别列 one-hot 身份上限（超过走 surrogate）")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="计算设备：auto（默认，有 GPU 自动用）/ cuda / cpu")
    parser.add_argument("--all-detectors", action="store_true",
                        help="启用全部 Stage2 检测器（重构/统计/低频拼写/关联/近似FD/近邻/聚类/形态离群）")
    parser.add_argument("--detectors", default=None,
                        help="逗号分隔指定启用的检测器，覆盖默认；可选: "
                             "reconstruction,statistical,categorical,association,fd,neighbor,clustering,pattern,numeric_format")
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "clean_mask": args.clean_mask,
        "stage1_errors": args.stage1_errors,
        "candidates_out": args.candidates_out,
        "combined_out": args.combined_out,
    }


def _resolve_config(args: argparse.Namespace) -> Stage2Config:
    cfg = Stage2Config()
    ov = _cli_overrides(args)
    dirty = args.input or cfg.paths.input_csv
    dp = cfg.set_paths_from_dataset(dirty, dataset=args.dataset, cli_overrides=ov)
    ensure_output_dirs(dp)

    if args.quantile is not None:
        cfg.scoring.quantile = args.quantile
    if args.margin is not None:
        cfg.scoring.margin = args.margin
    if args.min_predictability is not None:
        cfg.scoring.min_predictability = args.min_predictability
    if args.abs_prob_floor is not None:
        cfg.scoring.abs_prob_floor = args.abs_prob_floor
    if args.max_cells_per_row is not None:
        cfg.scoring.max_cells_per_row = args.max_cells_per_row
    if args.masked_inference:
        cfg.scoring.masked_inference = True
    if args.masked_inference_iters is not None:
        cfg.scoring.masked_inference_iters = args.masked_inference_iters
        if args.masked_inference_iters > 1:
            cfg.scoring.masked_inference = True
    if args.cdf_normalize:
        cfg.scoring.cdf_normalize = True
    if args.vocab_denoise:
        cfg.scoring.vocab_denoise = True
    if args.max_cardinality is not None:
        cfg.encoding.max_cardinality = args.max_cardinality
    if args.epochs is not None:
        cfg.model.epochs = args.epochs
    if args.device is not None:
        cfg.model.device = args.device

    _ALL = ["reconstruction", "statistical", "categorical",
            "association", "fd", "neighbor", "clustering", "pattern",
            "numeric_format"]
    if args.all_detectors:
        for name in _ALL:
            setattr(cfg.detectors, name, True)
    if args.detectors is not None:
        chosen = {s.strip() for s in args.detectors.split(",") if s.strip()}
        for name in _ALL:
            setattr(cfg.detectors, name, name in chosen)
    return cfg


def _maybe_build_llm():
    """构建 LLM 客户端供 FD 语义校验；无密钥或初始化失败时返回 None（退化为纯统计）。"""
    from stage_1.config import Stage1Config
    from stage_1.llm_rules import LLMClient

    s1_cfg = Stage1Config.resolve()
    if not s1_cfg.llm.api_key:
        print("[info] 未配置 LLM 密钥，FD 检测跳过语义校验（纯统计高召回）。")
        return None
    try:
        return LLMClient(s1_cfg)
    except Exception as exc:  # noqa: BLE001 - 初始化失败时安全退化
        print(f"[warn] LLM 初始化失败，FD 检测退化为纯统计：{exc}")
        return None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _resolve_config(args)

    df = read_table(cfg.paths.input_csv)
    print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

    mask_path = Path(cfg.paths.clean_mask)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"未找到干净掩码 {mask_path}，请先运行 Stage 1 生成 output/mask/{{dataset}}_clean_mask.csv。"
        )
    clean_mask = read_clean_mask(mask_path)
    row_clean = int(clean_mask.all(axis=1).sum())
    print(f"干净掩码: {int(clean_mask.values.sum())}/{clean_mask.size} 单元格干净, "
          f"{row_clean} 行整行干净（用于训练）")

    if cfg.scoring.vocab_denoise:
        from stage_2.vocab_denoise import denoise_clean_mask
        print("词表去污：编辑距离自过滤剔除漏报 typo...")
        clean_mask, vd_stats = denoise_clean_mask(
            df, clean_mask,
            encoding_cfg=cfg.encoding,
            max_exclude_ratio=cfg.scoring.vocab_denoise_ratio,
        )
        print(f"  剔除漏报 typo {vd_stats['excluded']} 格（因上限放弃 "
              f"{vd_stats['excluded_capped']}），干净格 {vd_stats['clean_before']} -> "
              f"{vd_stats['clean_after']}")

    model = build_model(**cfg.model_kwargs()) if cfg.detectors.reconstruction else None
    from paths.layout import resolve_dataset_paths
    from stage_2.coltypes import load_semantic_types
    rules_path = resolve_dataset_paths(cfg.paths.input_csv).rules
    semantic_types = load_semantic_types(rules_path)

    # FD 检测器默认做 LLM 语义校验：有可用密钥则注入 LLM，否则退化为纯统计高召回。
    llm = _maybe_build_llm() if cfg.detectors.fd else None

    print(f"启用检测器: {', '.join(cfg.detectors.enabled())}")
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
        llm=llm,
    )

    out = Path(cfg.paths.candidates_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    print(f"\nStage 2 检出 {len(candidates)} 个候选单元格 -> {out}")
    if len(candidates):
        if "detector" in candidates.columns:
            print("检测器分布:", dict(candidates["detector"].value_counts()))
        print("列分布:")
        print(candidates["column"].value_counts().to_string())

    s1_path = Path(cfg.paths.stage1_errors)
    if s1_path.exists():
        stage1 = read_table(s1_path)
        if "row_id" in stage1.columns:
            stage1["row_id"] = stage1["row_id"].astype(int)
    else:
        print(f"[warn] 未找到 Stage 1 错误文件 {s1_path}，仅融合 Stage 2 候选。")
        stage1 = pd.DataFrame()

    combined = merge_candidates(stage1, candidates)
    comb_out = Path(cfg.paths.combined_out)
    comb_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comb_out, index=False)
    print(f"\n证据融合 Stage1+Stage2 -> {len(combined)} 个候选单元格 -> {comb_out}")
    if not combined.empty:
        if "source" in combined.columns:
            print("来源分布:", dict(combined["source"].value_counts()))
        if "confidence_tier" in combined.columns:
            print("置信度分层:", dict(combined["confidence_tier"].value_counts()))


if __name__ == "__main__":
    main()
