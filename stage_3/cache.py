"""
Stage 3 LLM 响应缓存：prompt 文本 MD5 -> 原始响应字符串。

与 stage_1.rule_cache 同思路，避免对相同 prompt 重复调用 LLM，节省成本、保证可复现。
注意：若 prompt 模板或上下文构造变更，需手动删除缓存文件（或改 SYSTEM_PROMPT_VERSION）。

并发：Stage 3 用线程池并发调用 LLM，get/set 均加锁；写入去掉「每次全量重写」，
改为按批与结束时 flush（消除 O(N^2) 写放大，避免竞争损坏文件）。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading


class ResponseCache:
    """prompt -> LLM 响应 的本地 JSON 文件缓存（线程安全 + 延迟落盘）。"""

    def __init__(self, path: str = ".stage3_cache.json", flush_every: int = 50):
        self.path = path
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._dirty = 0                # 自上次落盘以来新增/更新的条目数
        self._flush_every = max(1, flush_every)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str):
        with self._lock:
            return self.cache.get(self._key(prompt))

    def set(self, prompt: str, response: str) -> None:
        with self._lock:
            self.cache[self._key(prompt)] = response
            self._dirty += 1
            if self._dirty >= self._flush_every:
                self._flush_locked()

    def flush(self) -> None:
        """把内存缓存落盘（调用方应在全部处理结束后调用一次）。"""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._dirty == 0:
            return
        # 原子写：先写临时文件再替换，避免进程中断时截断/损坏缓存文件
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        self._dirty = 0
