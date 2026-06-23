"""
Stage 2 多检测器框架（文档 §7-§14）。

每个检测器输入 (df, clean_mask, ctx)，输出统一的 list[CandidateError]。
检测器之间互不依赖，由 stage_2.score.run_stage2 按 DetectorConfig 编排，
再交 stage_2.fusion 做多证据融合。
"""

from __future__ import annotations
