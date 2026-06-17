"""
Stage 3 命令行入口：LLM 语义精检（确认 / 分类 / 建议修复）。

前置条件:
    先完成 Stage 1 与 Stage 2（产出 output/stage2/{dataset}_combined_candidates.csv 等）

运行命令（PowerShell，项目根目录）:

    python -m stage_3.cli --input data/hospital_dirty.csv --dry-run --limit 3
    python -m stage_3.cli --input data/hospital_dirty.csv
    python -m stage_3.cli --input data/hospital_dirty.csv --limit 20 --no-cache

产出（以 hospital 为例）:
    output/stage3/hospital_stage3_results.csv
    output/stage3/hospital_final_errors.csv

评估:
    python -m stage_3.evaluate --dirty data/hospital_dirty.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths.layout import ensure_output_dirs
from stage_3.cache import ResponseCache
from stage_3.config import Stage3Config
from stage_3.context import load_contexts
from stage_3.prompt import build_prompt
from stage_3.verifier import propagate_fix_mappings, verify_contexts

RESULT_COLUMNS = [
    "row_id", "column", "value", "prior_error_type", "prior_source",
    "is_error", "error_type", "confidence", "suggested_fix", "llm_reason",
]
FINAL_COLUMNS = ["row_id", "column", "error_type", "confidence", "suggested_fix"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 3: LLM semantic verification")
    p.add_argument("--input", default=None, help="原始脏表 CSV（data/{dataset}_dirty.csv）")
    p.add_argument("--dataset", default=None, help="显式指定数据集名")
    p.add_argument("--candidates", default=None, help="Stage1∪Stage2 候选 CSV")
    p.add_argument("--rules", default=None, help="Stage1 rules.json")
    p.add_argument("--results-out", default=None, help="逐格判定输出 CSV")
    p.add_argument("--final-out", default=None, help="最终确认错误输出 CSV")
    p.add_argument("--cache", default=None, help="LLM 响应缓存路径")
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 行分组（0=全部）")
    p.add_argument("--dry-run", action="store_true", help="只构造/打印 prompt，不调 LLM")
    p.add_argument("--no-cache", action="store_true", help="禁用响应缓存")
    p.add_argument("--reject-conf-threshold", type=float, default=None,
                   help="共识冲突候选被 LLM 否决所需的最低把握（默认 0.85）")
    p.add_argument("--min-avg-group", type=float, default=None,
                   help="探测 key 列的平均每取值行数门槛（默认 3.0）")
    p.add_argument("--min-dominance", type=float, default=None,
                   help="判定共识型列的组内主导占比门槛（默认 0.5）")
    p.add_argument("--min-lift", type=float, default=None,
                   help="组内占比相对全局基准的最小提升，排除类别不平衡伪共识（默认 0.15）")
    return p


def _cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "candidates": args.candidates,
        "rules": args.rules,
        "results_out": args.results_out,
        "final_out": args.final_out,
        "cache": args.cache,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Stage3Config.resolve()
    ov = _cli_overrides(args)
    dirty = args.input or cfg.paths.input_csv
    dp = cfg.set_paths_from_dataset(dirty, dataset=args.dataset, cli_overrides=ov)
    ensure_output_dirs(dp)

    if args.limit is not None:
        cfg.limit = args.limit
    if args.dry_run:
        cfg.dry_run = True
    if args.reject_conf_threshold is not None:
        cfg.reject_conf_threshold = args.reject_conf_threshold
    if args.min_avg_group is not None:
        cfg.min_avg_group = args.min_avg_group
    if args.min_dominance is not None:
        cfg.min_dominance = args.min_dominance
    if args.min_lift is not None:
        cfg.min_lift = args.min_lift

    df, contexts = load_contexts(
        cfg.paths.input_csv, cfg.paths.candidates, cfg.paths.rules,
        max_normal_samples=cfg.max_normal_samples,
        min_avg_group=cfg.min_avg_group,
        min_dominance=cfg.min_dominance,
        min_lift=cfg.min_lift,
    )
    if cfg.limit and cfg.limit > 0:
        contexts = contexts[:cfg.limit]
    n_cells = sum(len(c.suspects) for c in contexts)
    n_conflict = sum(
        1 for c in contexts for s in c.suspects if s.consensus_conflict
    )
    print(f"待精检: {len(contexts)} 行分组, {n_cells} 个可疑单元格 (数据集: {dp.dataset})")
    if n_conflict:
        print(f"  其中 {n_conflict} 个为跨行共识冲突格（受缺证据保护，"
              f"否决阈值={cfg.reject_conf_threshold}）")

    if cfg.dry_run:
        preview = min(3, len(contexts))
        print(f"\n[dry-run] 展示前 {preview} 行的 prompt:\n")
        for ctx in contexts[:preview]:
            print("=" * 60)
            print(build_prompt(ctx))
            print()
        return

    from stage_1.llm_rules import LLMClient
    llm = LLMClient(cfg.llm)
    cache = None if args.no_cache else ResponseCache(cfg.paths.cache)

    results = verify_contexts(
        contexts, llm, cache=cache,
        reject_conf_threshold=cfg.reject_conf_threshold,
    )
    results = propagate_fix_mappings(results)
    results_df = pd.DataFrame(results).reindex(columns=RESULT_COLUMNS)

    out = Path(cfg.paths.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n逐格判定已写入: {out} ({len(results_df)} 条)")

    confirmed = results_df[results_df["is_error"]].copy()
    final_df = confirmed.reindex(columns=FINAL_COLUMNS)
    final_out = Path(cfg.paths.final_errors_out)
    final_df.to_csv(final_out, index=False)
    print(f"最终确认错误: {len(final_df)} 个 -> {final_out}")

    rejected = len(results_df) - len(confirmed)
    print(f"被 LLM 否决（误报）: {rejected} 个")
    protected = int(results_df["llm_reason"].astype(str).str.startswith("[保护]").sum())
    if protected:
        print(f"缺证据保护：{protected} 个低把握否决被驳回，维持为错误（保住召回）")
    if len(confirmed):
        print("\n确认错误类型分布:")
        print(confirmed["error_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
