"""
Stage 3 LLM 响应缓存：prompt 文本 MD5 -> 原始响应字符串。

与 stage_1.rule_cache 同思路，避免对相同 prompt 重复调用 LLM，节省成本、保证可复现。
注意：若 prompt 模板或上下文构造变更，需手动删除缓存文件。
"""

from __future__ import annotations

import hashlib
import json
import os


class ResponseCache:
    """prompt -> LLM 响应 的本地 JSON 文件缓存。"""

    def __init__(self, path: str = ".stage3_cache.json"):
        self.path = path
        self.cache: dict[str, str] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str):
        return self.cache.get(self._key(prompt))

    def set(self, prompt: str, response: str) -> None:
        self.cache[self._key(prompt)] = response
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
