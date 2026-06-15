"""
Stage 3 prompt 构造：把单行上下文渲染为让 LLM 做语义精检的 JSON 任务。
"""

from __future__ import annotations

import json

from stage_3.context import RowContext


PROMPT_HEADER = """你是表格数据质量审核专家。下面给你一行数据的完整内容，以及前序检测器
（规则层 + 分布模型）标记出的若干"可疑单元格"。请结合整行上下文与同列正常样例，
逐个判断每个可疑单元格是否真的是错误，并给出类型、置信度与建议修复。

错误类型定义:
- MV : 缺失值（空/缺失哨兵）
- T  : 拼写错误（typo，与正确值仅少量字符差异）
- VAD: 违反跨列依赖（该值与同行其他列不一致，如 city/state/zip 不匹配）
- FI : 格式错误（格式/长度/取值范围不符合该列规范）
- OTHER: 其他错误
- NONE: 不是错误（前序为误报）

重要原则:
1. 利用整行上下文做跨列一致性判断。若某值虽被标记，但与同行其他列在语义上一致
   （例如某"州平均"编码的前缀与本行 State 一致，或某派生列由多列共同决定且取值合理），
   应判为 NONE（误报），不要盲从前序标记。
2. 正常样例代表该列常见的合法形态，可用于识别 typo / 格式问题。
3. 对确实是错误的，尽量给出最可能的正确值 suggested_fix；无法确定时置为 null。
4. confidence 取 0~1，表示"这是错误"的把握；判 NONE 时表示"这是误报"的把握。
"""

OUTPUT_SPEC = """## 输出（只输出 JSON，不要任何解释）
对每个可疑单元格给出一条判定，按给定顺序、列名一一对应:
{
  "judgments": [
    {
      "column": "列名",
      "is_error": true,
      "error_type": "MV|T|VAD|FI|OTHER|NONE",
      "confidence": 0.0,
      "suggested_fix": "最可能正确值，或 null",
      "reason": "简短理由"
    }
  ]
}"""


def _format_suspects(ctx: RowContext) -> list[dict]:
    """把可疑单元格整理成精简 JSON 结构。"""
    items = []
    for s in ctx.suspects:
        items.append({
            "column": s.column,
            "current_value": s.value,
            "semantic_type": s.semantic_type,
            "flagged_as": s.prior_error_type,
            "flagged_by": s.prior_source,
            "prior_reason": s.reason,
            "prior_suggested_fix": s.suggested_fix or None,
            "normal_samples": list(s.normal_samples),
        })
    return items


def build_prompt(ctx: RowContext) -> str:
    """渲染单行的精检 prompt。"""
    row_json = json.dumps(ctx.row_values, ensure_ascii=False, indent=2, default=str)
    suspects_json = json.dumps(_format_suspects(ctx), ensure_ascii=False, indent=2, default=str)
    return (
        f"{PROMPT_HEADER}\n"
        f"## 整行数据 (row_id={ctx.row_id})\n{row_json}\n\n"
        f"## 可疑单元格（共 {len(ctx.suspects)} 个）\n{suspects_json}\n\n"
        f"{OUTPUT_SPEC}"
    )
