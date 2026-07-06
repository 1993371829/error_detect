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
from typing import Optional

from stage_1.config import Stage1Config
from stage_1.llm_usage import record_llm_usage


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

## 列逻辑类型 logical_type（借鉴语义类型校验）
除规则外，请额外推断该列在语义上应有的"逻辑类型" logical_type, 取值之一:
- bool:   只应取真假两态（如 yes/no、true/false、0/1、Y/N）
- int:    只应是整数
- float:  只应是数值（含小数）
- date:   只应是日期
- categorical: 有限枚举类别（非数值）
- string: 自由文本/标识符，不做类型约束
规则: 仅当你**很有把握**该列应是 bool/int/float/date 时才给出对应类型；
任何不确定、混合或自由文本一律用 string 或 categorical（这两者不会触发类型校验）。
保守为先：宁可填 string 也不要误判类型导致大量误报。

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
  "logical_type": "bool/int/float/date/categorical/string 之一",
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
        self.max_tokens = config.llm.max_tokens
        self.max_retries = config.llm.max_retries
        self.enable_thinking = config.llm.enable_thinking

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """发送 prompt 并返回模型回复文本（强制 JSON 格式）。

        Args:
            system: 可选 system 消息。把稳定的静态提示词放入 system，可命中服务商前缀缓存，
                    显著降低 prompt 计费（判定输出不变）。
            enable_thinking: 覆盖本次调用的思考模式；None 时回退到客户端配置 self.enable_thinking。
                    最终非 None 时经 extra_body 传给 qwen3 等推理模型（False=关思考省 token）。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        if self.max_tokens and self.max_tokens > 0:
            kwargs["max_tokens"] = self.max_tokens  # 截断保护
        think = enable_thinking if enable_thinking is not None else self.enable_thinking
        if think is not None:
            kwargs["extra_body"] = {"enable_thinking": bool(think)}
        resp = self.client.chat.completions.create(**kwargs)
        record_llm_usage(getattr(resp, "usage", None))
        return resp.choices[0].message.content


def _extract_json(raw: Optional[str]) -> Optional[dict]:
    """
    宽松解析 LLM 输出为 dict：
        1. 直接 json.loads；
        2. 去掉 ```json / ``` markdown 围栏后再试；
        3. 回退取首个 '{' 到末个 '}' 的子串再试。

    全部失败返回 None（不抛异常）。
    """
    if not raw:
        return None
    candidates = []
    text = raw.strip()
    candidates.append(text)

    # 去掉 markdown 代码围栏（```json ... ``` 或 ``` ... ```）
    if text.startswith("```"):
        body = text[3:]
        if body[:4].lower() == "json":
            body = body[4:]
        body = body.strip()
        if body.endswith("```"):
            body = body[:-3].strip()
        candidates.append(body)

    # 取首个 '{' 到末个 '}' 的子串（救前后多余文本 / 轻度截断）
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi != -1 and hi > lo:
        candidates.append(text[lo:hi + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def complete_json(llm: "LLMClient", prompt: str, *, label: str = "") -> Optional[dict]:
    """
    通用的健壮 JSON 补全：截断保护(complete 的 max_tokens) + 宽松解析(_extract_json)
    + 失败重试(llm.max_retries)。供规则归纳/标准化/FD 语义校验等所有 Stage 1 LLM 调用复用。

    Returns:
        解析成功的 dict；全部尝试失败返回 None（不抛异常、不中断流水线）。
    """
    attempts = 1 + max(0, getattr(llm, "max_retries", 0))
    tag = f"({label}) " if label else ""
    for _ in range(attempts):
        try:
            raw = llm.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - 接口/网络异常计入重试，不中断流水线
            print(f"[llm-warn] {tag}LLM 调用异常，重试: {exc}")
            continue
        data = _extract_json(raw)
        if data is not None:
            return data
    return None


def extract_rules_for_column(llm: LLMClient, profile: dict) -> dict:
    """
    对单列画像调用 LLM，解析返回的 JSON 规则。

    经 complete_json 做截断保护 + 宽松解析 + 重试；全部失败时返回空 rules 列表并打印
    警告，不中断流水线。
    """
    prompt = RULE_EXTRACTION_PROMPT.format(
        profile=json.dumps(profile, ensure_ascii=False, indent=2)
    )
    data = complete_json(llm, prompt, label=f"列 {profile['name']}")
    if data is not None:
        return data
    print(f"[warn] 列 {profile['name']} 的 LLM 输出无法解析为 JSON,跳过")
    return {"column": profile["name"], "rules": []}
