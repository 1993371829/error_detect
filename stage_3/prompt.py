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
- DMV: 伪缺失值（非空但语义为缺失/未知/不适用，如 "?"、"unknown"、"missing"、占位数字 "9999"）
- T  : 拼写错误（typo，与正确值仅少量字符差异）
- VAD: 违反跨列依赖（该值与同行其他列不一致，如 city/state/zip 不匹配）
- FI : 格式错误（格式/长度/取值范围/逻辑类型不符合该列规范，如本应是 bool/数值/日期却不符）
- OTHER: 其他错误
- NONE: 不是错误（前序为误报）

重要原则:
1. 利用整行上下文做跨列一致性判断。若某值虽被标记，但与同行其他列在语义上一致
   （例如某"州平均"编码的前缀与本行 State 一致，或某派生列由多列共同决定且取值合理），
   应判为 NONE（误报），不要盲从前序标记。
2. 正常样例代表该列常见的合法形态，可用于识别 typo / 格式问题。
   若提供 column_stats（该列统计画像），请据此基于"分布"而非单值判断:
   - value_frequencies 给出高频合法值及出现次数；偏离主流且低频的值更可能是 string outlier / typo。
   - dominant_patterns 是该列主导字符形态（d=数字, L=大写, l=小写）；与主导模式不符的值更可能是格式错误(FI)。
   - numeric_range 给出数值列的 min/max/mean；明显超出范围的值更可能是数值离群(FI/OTHER)。
   - null_rate 高时，"?/unknown/missing/占位数字"等更可能是伪缺失值(DMV)。
   - 不一致表示（重要）：当 value_frequencies 显示同一概念存在多种写法
     （单位/大小写/缩写/符号后缀不一致，如 "12.0 oz." / "12.0 ounce" / "12.0 OZ." 实为同义，
     或 "0.09%" 多了百分号），即便该非规范写法是高频/多数派，也应判为格式错误(FI)，
     suggested_fix 取该列语义最规范、最统一的写法（如 "12.0 oz"、"0.09"）。
     不要因为"该写法很常见"就判 NONE——这正是需要被标准化纠正的系统性错误。
3. 若提供 cross_row_consensus（按某 key 列分组后，同 key 其他行对该列的取值分布）:
   - 当本格取值与同 key 多数值不一致（conflicts_with_majority=true）且多数值占比较高时，
     即便本值"看起来格式合法"，通常也是与其他记录冲突的错误（应判为错误，类型 VAD/OTHER，
     suggested_fix 取 majority_value）。这类错误无法只看单行判定，务必依据共识证据，
     不要因"格式正常"就轻易判 NONE。
   - 当本格取值与同 key 多数值一致时，倾向判 NONE。
   - model_anomaly_score 越高，表示分布模型越认为本值可疑，可作为参考。
4. 若提供 multi_detector_evidence（多检测器证据融合）:
   - detectors 列出命中本格的检测器；命中越多、suspicion_score 越高、confidence_tier 越高，
     越可能是真错误。多个独立检测器一致指向同一格时应提高判定为错误的把握。
   - independent_family_count 为命中的**独立证据族**数（规则/模型/统计/近邻/模式/文本），仅供参考：
     多族一致**不必然**是真错——高基数/自由格式列上多个无监督检测器常对同一正常值系统性同时误报。
     务必回到该值本身是否语义/事实错误来判断，不能仅因族数高就判为错误。
   - candidate_fixes 汇总各检测器给出的候选修复值，可作为 suggested_fix 的参考。
5. 对确实是错误的，请给出标准化的正确值 suggested_fix（old->new 映射思路）:
   - 优先映射到该列 value_frequencies / normal_samples 中已存在的规范表示
     （如 "English"->"eng"、"NY"->"New York" 取该列的主流写法）。
   - 同一脏值在该列应映射到同一规范值，保持一致。
   - DMV / MV 这类缺失，若无法恢复真实值，suggested_fix 置为 null。
   - 无法确定正确值时置为 null。
6. confidence 取 0~1，表示"这是错误"的把握；判 NONE 时表示"这是误报"的把握。
   当依据充分（如明显冲突或明显 typo）请给出较高 confidence。
"""

OUTPUT_SPEC = """## 输出（只输出 JSON，不要任何解释）
对每个可疑单元格给出一条判定，按给定顺序、列名一一对应:
{
  "judgments": [
    {
      "column": "列名",
      "is_error": true,
      "error_type": "MV|DMV|T|VAD|FI|OTHER|NONE",
      "confidence": 0.0,
      "suggested_fix": "标准化后的正确值，或 null",
      "reason": "简短理由（说明判定依据，非推理模型据此稳定判定）"
    }
  ]
}"""

# 静态系统提示词：任务说明 + 输出规范，跨行完全一致 -> 放入 system 消息命中前缀缓存。
# 版本号：模板变更时改此值，使响应缓存键失效，避免复用旧 prompt 的结果。
SYSTEM_PROMPT_VERSION = "v3"
SYSTEM_PROMPT = f"{PROMPT_HEADER}\n{OUTPUT_SPEC}"


def _format_suspects(ctx: RowContext) -> list[dict]:
    """把可疑单元格整理成精简 JSON 结构。"""
    items = []
    for s in ctx.suspects:
        item = {
            "column": s.column,
            "current_value": s.value,
            "semantic_type": s.semantic_type,
            "flagged_as": s.prior_error_type,
            "flagged_by": s.prior_source,
            "prior_reason": s.reason,
            "prior_suggested_fix": s.suggested_fix or None,
            "normal_samples": list(s.normal_samples),
        }
        if s.anomaly_score is not None:
            item["model_anomaly_score"] = round(float(s.anomaly_score), 3)
        if s.confidence_tier:
            fusion = {
                "detectors": s.detectors,
                "confidence_tier": s.confidence_tier,
            }
            if s.suspicion_score is not None:
                fusion["suspicion_score"] = round(float(s.suspicion_score), 3)
            if s.family_count:
                fusion["independent_family_count"] = s.family_count
            if s.fused_evidence:
                fusion["evidence"] = s.fused_evidence
            if s.candidate_fixes:
                fusion["candidate_fixes"] = s.candidate_fixes
            item["multi_detector_evidence"] = fusion
        if s.column_stats:
            cs = s.column_stats
            stats = {
                "null_rate": cs.get("null_rate"),
                "distinct_count": cs.get("distinct_count"),
                "value_frequencies": cs.get("top_values"),  # [[值, 出现次数], ...]
                "dominant_patterns": cs.get("dominant_patterns"),
            }
            if cs.get("numeric_range"):
                stats["numeric_range"] = cs["numeric_range"]
            item["column_stats"] = stats
        if s.consensus:
            c = s.consensus
            item["cross_row_consensus"] = {
                "grouped_by": c["key_column"],
                "key_value": c["key_value"],
                "group_size": c["group_size"],
                "majority_value": c["majority_value"],
                "majority_share": c["majority_share"],
                "current_value_share": c["current_share"],
                "conflicts_with_majority": c["is_conflict"],
            }
        items.append(item)
    return items


def build_user_prompt(ctx: RowContext) -> str:
    """渲染单行的动态 user 部分（整行数据 + 可疑单元格），静态说明见 SYSTEM_PROMPT。"""
    row_json = json.dumps(ctx.row_values, ensure_ascii=False, indent=2, default=str)
    suspects_json = json.dumps(_format_suspects(ctx), ensure_ascii=False, indent=2, default=str)
    return (
        f"## 整行数据 (row_id={ctx.row_id})\n{row_json}\n\n"
        f"## 可疑单元格（共 {len(ctx.suspects)} 个）\n{suspects_json}"
    )


def build_prompt(ctx: RowContext) -> str:
    """完整 prompt（system + user 拼接），供 dry-run 预览与向后兼容。"""
    return f"{SYSTEM_PROMPT}\n{build_user_prompt(ctx)}"
