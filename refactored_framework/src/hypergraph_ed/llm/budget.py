from __future__ import annotations

import time
import uuid
from pathlib import Path

from ..artifacts import atomic_json, read_json


LIMITS = {"structure": (6, 40000), "audit": (4, 20000), "labels": (8, 30000), "retry": (2, 10000)}


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Write-ahead reservations; interrupted requests remain fully charged.

    A run-level OS lock serializes access. Do not share this object outside that lock.
    """
    def __init__(self, path: Path, identity: str):
        self.path = path
        if path.exists():
            self.data = read_json(path)
            if self.data["identity"] != identity:
                raise ValueError("budget identity mismatch; use a new output directory")
        else:
            self.data = {"identity": identity, "events": [], "cache_hits": 0, "pseudo_judgments": 0}
            self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def used(self, phase: str | None = None) -> tuple[int, int]:
        events = [x for x in self.data["events"] if phase is None or x["phase"] == phase]
        return len(events), sum(x["charged_tokens"] for x in events)

    def reserve(self, phase: str, input_tokens: int, output_tokens: int, judgments: int = 0, judgment_cap: int = 128) -> str:
        if phase not in LIMITS or input_tokens < 0 or output_tokens <= 0:
            raise ValueError("invalid request reservation")
        amount = input_tokens + output_tokens
        if judgments < 0 or self.data["pseudo_judgments"] + judgments > min(128,judgment_cap):
            raise BudgetExceeded("pseudo-label judgment budget exhausted")
        calls, tokens = self.used(phase)
        total_calls, total_tokens = self.used()
        max_calls, max_tokens = LIMITS[phase]
        if calls >= max_calls or tokens + amount > max_tokens or total_calls >= 20 or total_tokens + amount > 100000:
            raise BudgetExceeded(f"request cannot fit {phase} budget")
        request_id = uuid.uuid4().hex
        self.data["pseudo_judgments"] += judgments
        self.data["events"].append({"id": request_id, "phase": phase, "input_reserved": input_tokens, "output_reserved": output_tokens, "charged_tokens": amount, "judgments": judgments, "status": "pending", "started": time.time()})
        self.save()
        return request_id

    def settle(self, request_id: str, input_tokens: int | None, output_tokens: int | None, status: str, seconds: float) -> None:
        event = next(x for x in self.data["events"] if x["id"] == request_id)
        if event["status"] != "pending":
            raise ValueError("reservation already settled")
        if input_tokens is not None and output_tokens is not None:
            if input_tokens < 0 or output_tokens < 0:
                raise ValueError("negative reported usage")
            actual = input_tokens + output_tokens
            if actual > event["charged_tokens"]:
                event.update(status="accounting_mismatch", charged_tokens=actual, actual_input=input_tokens, actual_output=output_tokens)
                self.save()
                raise BudgetExceeded("provider usage exceeds tokenizer reservation; no further requests allowed")
            event.update(charged_tokens=actual, actual_input=input_tokens, actual_output=output_tokens)
        event.update(status=status, seconds=seconds)
        self.save()

    def claim_labels(self, number: int, maximum: int = 128) -> bool:
        if number < 0 or self.data["pseudo_judgments"] + number > min(maximum, 128):
            return False
        self.data["pseudo_judgments"] += number
        self.save()
        return True
