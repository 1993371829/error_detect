# 项目记忆 / 架构与变更记录

> 本文件是 `error_dect` 的项目记忆（按全局规则统一维护于 `docs/project_context.md`）。
> 记录架构、关键约定、重要逻辑路径与踩坑经验。新变更追加在“变更日志”。

## 1. 项目概览

三阶段漏斗式表格数据错误检测：

- **Stage 1 规则层**（高精度）：规则 + 统计 + LLM 归纳 + 高精度检测器。
- **Stage 2 多检测器候选层**（高召回）：多检测器并行 → 证据融合 → 置信度分层。
- **Stage 3 LLM 精检层**：基于证据包的语义确认 / 分类 / 误报过滤 / 标准化修复。

运行顺序（必须按序）：

```powershell
python main.py --input data/<ds>_dirty.csv                       # Stage 1
python -m stage_2.cli --input data/<ds>_dirty.csv [--all-detectors]  # Stage 2
python -m stage_3.cli --input data/<ds>_dirty.csv [--min-tier mid]   # Stage 3
```

## 2. Stage 2 多检测器 + 证据融合架构（本次升级核心）

### 2.1 统一数据结构（`stage_2/schema.py`）
- `CandidateError`：单检测器对某格的一条判定（row_id/column/value/detector/error_type/score/evidence/suggested_fix/metadata）。
- `SuspiciousCell`：融合后单元格结果（suspicion_score/confidence_tier/evidence_list/candidate_fixes）。
- `candidates_to_frame()`：统一 schema 落表（含 `detector`、`subtype` 列）。

### 2.2 列类型/语义访问层（`stage_2/coltypes.py`）
- `infer_column_kinds()`：numeric/categorical/id/highcard/empty，口径与 `encoding.py` 对齐。
- `robust_stats()`：median/MAD(×1.4826)/IQR，含退化兜底。
- `parse_numeric()` 复用 `encoding._extract_number`（单位感知，`97%`/`33 patients` 视为数值）。
- `load_semantic_types()`：读 `rules.json` 的 `semantic_type`。

### 2.3 检测器（`stage_2/detectors/`，统一返回 `list[CandidateError]`）
| 检测器 | 文件 | 类型 | 说明 |
|---|---|---|---|
| reconstruction | `reconstruction.py` | DIST | 包装现有 `ConditionalPredictor`（条件预测打分），默认开启 |
| statistical | `statistical.py` | DIST | 数值 Robust-Z + IQR；日期解析后 Robust-Z |
| categorical | `categorical.py` | T/FI | 低频值 + rapidfuzz 拼写相似 + 罕见字符模式 |
| association_rule | `association_rule.py` | VAD | 单前件关联规则 (A=a)=>(B=b)，support/confidence/lift 门控 |
| approx_fd | `fd_detector.py` | VAD | 复用 `stage_1.fd_detect` 统计挖掘；**FD 已从 Stage 1 迁入此处**，默认开 LLM 语义校验（`semantic_check=True`），无 LLM 时退化为纯统计高召回 |
| neighbor_consistency | `neighbor_consistency.py` | DIST/VAD | sklearn KNN 在干净行建邻域，数值偏离/类别多数 |
| clustering | `clustering.py` | DIST | LOF 行级离群 + 逐列 robust-z/低频定位（权重低） |

`DetectorContext`（`detectors/base.py`）携带 kinds/semantic_types/encoder/x_all/row_clean 供各检测器复用，避免重复计算。

### 2.4 证据融合（`stage_2/fusion.py`）
- 归一化：软检测器（reconstruction/statistical/neighbor/clustering）按 detector 分位归一化到 [0,1]；硬/规则检测器用其置信度（NaN/≤0 → 默认 0.9）。
- 权重 `_WEIGHTS`（文档 §15.3）：strong_rule=1.0、approx_fd/fd=0.9、reconstruction=0.85、association/typo=0.8、neighbor=0.75、statistical/format_cluster=0.6、clustering=0.4。
- 融合公式：`S = 1 - ∏(1 - w_d * s_d)`，每检测器取最强证据。
- 分层：high≥0.85 / mid≥0.6 / low（<0.4 也记 low，不丢弃以保召回）。
- 输出保留 Stage3 所需原字段（error_type/violated_rule/reason/suggested_fix/anomaly_score/subtype/source）+ 新增 suspicion_score/confidence_tier/evidence/candidate_fixes/detectors。
- `io_utils.merge_candidates()` 已改为调用 `fuse_candidates()`。

### 2.5 配置与开关（`stage_2/config.py: DetectorConfig`）
- 默认开核心高召回检测器：`reconstruction/statistical/categorical/neighbor/fd=True`；`association/clustering=False`（误报较高，按需开）。
- CLI：`--all-detectors` 一键全开；`--detectors a,b,c` 选择子集。
- FD 语义校验所需 LLM 由 `stage_2/cli.py:_maybe_build_llm()` 注入（复用 Stage1 `.env` 密钥），无密钥则自动退化。

## 3. Stage 1 检测器（高精度/确定性）
职责边界：Stage 1 只保留高精度规则与确定性检测；**Typo / 主导格式(format_outlier) / 近似函数依赖(FD) 已迁出并并入 Stage 2 多检测器层**，消除重复检测。
| 检测器 | 文件 | 默认 | 说明 |
|---|---|---|---|
| MV/DMV/标准化/泄漏/重复值 | `*_detect.py` | 开 | 缺失值、伪缺失、不一致表示、元数据泄漏、整值重复拼接 |
| 跨列对调 | `xcol_detect.py` | 开(需LLM) | 全名↔缩写列对 LLM 语义确认 + 两列对调标记 |
| 硬范围规则 | `range_detect.py` | 开 | 按 semantic_type 关键词施加 age/percentage/price/lat/lon/year 等硬范围 → FI/range_error |
| 极端统计兜底 | `detect_statistical`（复用 stage_2） | 开 | 数值/日期列 Robust-Z(>6)+IQR(k=4.5) 双判据，只捞确凿极端值 → FI/extreme_outlier；温和离群交 Stage 2 |
| 孤立森林 | `iforest_detect.py` | 关 | 数值列联合 IsolationForest top 分位 + 逐列 robust-z 定位 |
| LLM 跨列算术 | `arith_rules.py` | 关 | LLM 归纳算术约束 → **AST 安全求值**（白名单节点/函数）→ support/violation 验证 |
配置：`stage_1/config.py` 的 `execution.range_check/iforest/arith/statistic_extreme`；执行接入 `executor.py`（列循环后全局检测）。
- `typo_detect.py` 仅作为 `levenshtein` 工具保留（被 stage_2 vocab_denoise/diagnose_mask 复用）；`fmt_detect.py` 已删除。

**安全要点**：`arith_rules.compile_arith` 用 `ast.walk` 白名单校验，仅允许数字/列名/`+ - * / % **`/比较/and-or-not/`abs|min|max|round`，其余节点抛 `UnsafeExpression`，杜绝注入。

## 4. Stage 3 升级
- 证据包：`SuspectCell` 增 detectors/suspicion_score/confidence_tier/fused_evidence/candidate_fixes；prompt 注入 `multi_detector_evidence`。
- 分层过滤：`load_contexts(min_tier=...)` + `--min-tier {low,mid,high}`，只精检 ≥ 该层级的候选。
- 修复来源：results/final 增 `fix_source`（llm/consensus/prior/prior_rule/propagated）与 `fix_confidence`。
- 自动确认：`AUTO_CONFIRM_RULES={"duplicate_value"}`；`format_outlier` 已随检测器迁出，不再自动确认（改走融合 + LLM 复核）。

## 5. 评估与消融
- `stage_2.evaluate`：原 cell-level + 分组 + 新增 **row-level** P/R/F1。
- `stage_3.evaluate`：原前后对比 + 新增 **correction-level** 修复准确率 + **按 error_type 分项**。
- `stage_2.ablation`：一次训练 + 各检测器各跑一次，按子集组合对比合并集 P/R/F1（含留一法）。

## 6. 关键约定 / 踩坑
- 环境：Windows + Anaconda Python 3.9；PowerShell 下 `head`/`2>nul` 不可用，列目录用 `cmd /c "dir /b ..."`。
- 新文件必须 `from __future__ import annotations`（3.9 才能用 `X | None` 注解）。
- CSV 读写统一 `dtype=str, keep_default_na=False, na_values=[""]`（`io_utils.read_table`）。
- 融合中 Stage1 空 `confidence` 读为 NaN，**必须**兜底为默认权重，否则误降为 low 分层（已修复）。
- Stage 3 `context.py` 用 `.get()` 读候选列，新增列向后兼容；务必保留 anomaly_score/subtype 以维持共识保护。
- 依赖：新增 `scikit-learn>=1.2`、`rapidfuzz>=3.0`（见 requirements.txt）。

## 7. 实测（默认检测器集 reconstruction+statistical+categorical+neighbor+fd）
合并集(Stage1∪Stage2) cell-level / row-level（vs clean）：
- hospital：cell P=0.419 R=0.893 F1=0.570；row R=0.951。
- flights：cell P=0.763 R=0.794 F1=0.778；row R=0.809。
- beers：cell P=0.564 R=0.998 F1=0.721；row R=1.000。
- FD 语义校验生效示例：hospital 11→9、beers 4→3（否决单州/同名城市等伪依赖）。
- 结论：多检测器以 FP 换高召回，精度由 Stage 3 LLM 兑现；`association/clustering` 默认关闭控误报，按需 `--all-detectors`/`--min-tier`。

## 变更日志
- 2026-06：落地文档「多检测器 + 证据融合」框架。新增 stage_2 schema/coltypes/fusion/ablation 与 detectors 包（7 检测器）；新增 stage_1 range/iforest/arith；Stage3 证据包/分层/修复来源；评估扩展 row/correction/by-type。
- 2026-06：去重构重构。Stage 1 移除 Typo/主导格式/FD（并入 Stage 2），删除 `fmt_detect.py`；新增极端统计兜底(robust_z>6/IQR k=4.5)。Stage 2 默认开 statistical/categorical/neighbor/fd，approx_fd 默认 LLM 语义校验（cli 注入 LLM）。Stage 3 `AUTO_CONFIRM_RULES` 去除 format_outlier。
