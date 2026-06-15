"""
Step 2: LLM 规则归纳。

每列调用一次 LLM，输入 Step 1 的列画像，输出结构化 JSON 规则。

支持的规则类型 (type):
    - not_null:     不允许空值
    - regex:        正则格式校验
    - value_set:    枚举合法取值
    - length:       字符串长度约束
    - numeric_range: 数值范围约束

错误类型 (error_type):
    - MV:    Missing Value（缺失值）
    - FI:    Formatting Issue（格式/长度/范围问题）
    - VAD:   Value At Domain（不在合法取值集合）
    - OTHER: 其他
"""

from __future__ import annotations

import json

from stage_1.config import Stage1Config


# LLM 提示词模板：要求输出 JSON，并强调不要过度拟合脏数据
RULE_EXTRACTION_PROMPT = """你是数据质量专家。根据一张表格某一列的数据画像，
注意: 表格数据本身可能包含错误，不要"过度拟合"生成过于严格的规则，
宁可偏向宽松，把可疑值留给后续检测。不要为少量噪声异常值放宽规则。

## 列画像
{profile}

画像中除 sample_values / observed_patterns 外，还请参考:
- charset_info: 全列字符集标志与 special_chars（列中出现过的特殊字符）
- edge_samples: 含特殊字符、最长、最短的边缘样本
- pattern_coverage: distinct_pattern_count 与 top5_coverage_rate

## 任务
推断这一列应遵守的规则，只能使用以下规则类型:
- not_null: 不允许空值，spec 形如 {{}}
- regex: 标准格式，spec 形如 {{"pattern": "^\\\\d{{5}}$"}}
- value_set: 有限分类字段的合法取值，spec 形如 {{"values": ["TX","CA",...]}}
  当 unique_count 较小(如 < 50)，且明显是枚举类型时使用
- length: 长度约束，spec 形如 {{"fixed": 5}} 或 {{"min": 1, "max": 50}}
- numeric_range: 数值范围，spec 形如 {{"min": 0, "max": 150}}
  仅当 numeric_stats 不为 null 时使用

每条规则要标注对应的错误类型 error_type, 取值: MV / FI / VAD / OTHER
- not_null 仅表达语义约束（该列不应为空），实际 MV 检测由系统独立扫描负责
- 格式/长度/数值范围违反 -> FI
- 不在取值集合合法 -> FI

缺失值说明:
- 系统会自动将空串、NaN、empty 等缺失哨兵识别为 MV，无需在 regex 中写 |empty
- 若列不应为空，请输出 not_null 规则作语义记录；格式规则(regex/length 等)只描述非空值的合法形态

## 重要约束
1. sample_values 和 observed_patterns 只是高频样本,不代表全部。
   字符集请以 charset_info 为准,不要把 charset_info 里出现过的字符
   (尤其 special_chars 里的 & . - / 等)排除在正则之外。

2. 根据 semantic_type 决定是否用字符级正则:
   - 结构化字段(zip_code/phone/date/email/各类 code):可以用严格正则。
   - 自由文本字段(name/address/city/free_text):不要用 ^[a-z ]+$ 这类
     限定字符集的正则。这类字段允许大小写、数字、标点等任意可见字符。
     如果 distinct_pattern_count 很大或 top5_coverage_rate 较低,
     说明结构松散,应避免 regex,改用 length 约束或不加格式规则。

3. 宁可规则偏松也不要偏严: 漏报由后续模型补,但误报会污染结果。

## 输出 JSON (只输出 JSON, 不要任何解释)
{{
  "column": "列名",
  "semantic_type": "推断的语义类型(如 zip_code/state_code/date/email/name/free_text)",
  "confidence": 0.0,
  "rules": [
    {{
      "type": "regex",
      "spec": {{"pattern": "^\\\\d{{5}}$"}},
      "error_type": "FI",
      "reason": "95% 的值是 5 位数字"
    }}
  ]
}}"""


class LLMClient:
    """OpenAI 兼容接口的 LLM 客户端封装。"""

    def __init__(self, config: Stage1Config):
        from openai import OpenAI

        if not config.llm.api_key:
            raise ValueError(
                "LLM API key not configured. "
                "Copy .env.example to .env in the project root and set LLM_API_KEY, "
                "or export LLM_API_KEY in your shell."
            )
        self.client = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        self.model = config.llm.model
        self.temperature = config.llm.temperature

    def complete(self, prompt: str) -> str:
        """发送 prompt 并返回模型回复文本（强制 JSON 格式）。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


def extract_rules_for_column(llm: LLMClient, profile: dict) -> dict:
    """
    对单列画像调用 LLM，解析返回的 JSON 规则。

    若 JSON 解析失败，返回空 rules 列表并打印警告，不中断流水线。
    """
    prompt = RULE_EXTRACTION_PROMPT.format(
        profile=json.dumps(profile, ensure_ascii=False, indent=2)
    )
    raw = llm.complete(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[warn] 列 {profile['name']} 的 LLM 输出无法解析为 JSON,跳过")
        return {"column": profile["name"], "rules": []}
