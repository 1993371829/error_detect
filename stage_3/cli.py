"""
Stage 3 命令行入口：LLM 语义精检（确认 / 分类 / 建议修复）。

前置条件:
    pip install -r requirements.txt
    配置 LLM：.env 中填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
    先完成 Stage 1 与 Stage 2，确保存在:
        data/combined_candidates.csv
        data/hospital_rules.json
        data/hospital_dirty.csv

运行命令（在项目根目录 d:\\study\\error_dect 下执行）:

    # hospital 全流程 — Stage 3（建议先 dry-run 确认 prompt）
    python -m stage_3.cli --dry-run --limit 3

    # 正式全量精检
    python -m stage_3.cli

    # 小规模试跑（省钱）
    python -m stage_3.cli --limit 20

    # 显式指定输入/输出路径
    python -m stage_3.cli --input data/hospital_dirty.csv --candidates data/combined_candidates.csv --rules data/hospital_rules.json --results-out data/stage3_results.csv --final-out data/final_errors.csv

    # 禁用 LLM 响应缓存
    python -m stage_3.cli --no-cache

评估（需 data/hospital_clean.csv）:
    python -m stage_3.evaluate

产出:
    data/stage3_results.csv  逐格判定明细（is_error / confidence / suggested_fix / llm_reason）
    data/final_errors.csv    最终确认错误 (row_id, column, error_type, confidence, suggested_fix)

全流程回顾:
    python main.py --input data/hospital_dirty.csv --output data/hospital_errors.csv --rules-out data/hospital_rules.json
    python -m stage_2.cli
    python -m stage_3.cli
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage_3.cache import ResponseCache
from stage_3.config import Stage3Config
from stage_3.context import load_contexts
from stage_3.prompt import build_prompt
from stage_3.verifier import verify_contexts

RESULT_COLUMNS = [
    "row_id", "column", "value", "prior_error_type", "prior_source",
    "is_error", "error_type", "confidence", "suggested_fix", "llm_reason",
]
FINAL_COLUMNS = ["row_id", "column", "error_type", "confidence", "suggested_fix"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 3: LLM semantic verification")
    p.add_argument("--input", default=None, help="原始脏表 CSV")
    p.add_argument("--candidates", default=None, help="Stage1∪Stage2 候选 CSV")
    p.add_argument("--rules", default=None, help="Stage1 rules.json")
    p.add_argument("--results-out", default=None, help="逐格判定输出 CSV")
    p.add_argument("--final-out", default=None, help="最终确认错误输出 CSV")
    p.add_argument("--cache", default=None, help="LLM 响应缓存路径")
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 行分组（0=全部）")
    p.add_argument("--dry-run", action="store_true", help="只构造/打印 prompt，不调 LLM")
    p.add_argument("--no-cache", action="store_true", help="禁用响应缓存")
    return p


def _apply_overrides(cfg: Stage3Config, args: argparse.Namespace) -> None:
    if args.input:
        cfg.paths.input_csv = args.input
    if args.candidates:
        cfg.paths.candidates = args.candidates
    if args.rules:
        cfg.paths.rules = args.rules
    if args.results_out:
        cfg.paths.results_out = args.results_out
    if args.final_out:
        cfg.paths.final_errors_out = args.final_out
    if args.cache:
        cfg.paths.cache = args.cache
    if args.limit is not None:
        cfg.limit = args.limit
    if args.dry_run:
        cfg.dry_run = True


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Stage3Config.resolve()
    _apply_overrides(cfg, args)

    df, contexts = load_contexts(
        cfg.paths.input_csv, cfg.paths.candidates, cfg.paths.rules,
        max_normal_samples=cfg.max_normal_samples,
    )
    if cfg.limit and cfg.limit > 0:
        contexts = contexts[:cfg.limit]
    n_cells = sum(len(c.suspects) for c in contexts)
    print(f"待精检: {len(contexts)} 行分组, {n_cells} 个可疑单元格")

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

    results = verify_contexts(contexts, llm, cache=cache)
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
    if len(confirmed):
        print("\n确认错误类型分布:")
        print(confirmed["error_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
