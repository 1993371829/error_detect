import json

import pytest

from hypergraph_ed.llm import BudgetLedger, BudgetExceeded, BudgetedClient
from hypergraph_ed.config import LLMConfig


def test_pending_requests_stay_charged_and_cannot_reset(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path,"run")
    ledger.reserve("structure",100,1000)
    restored = BudgetLedger(path,"run")
    assert restored.used() == (1,1100)
    with pytest.raises(ValueError):
        BudgetLedger(path,"different input")


def test_limits_and_actual_usage(tmp_path):
    ledger = BudgetLedger(tmp_path/"b.json","a")
    for _ in range(6):
        rid = ledger.reserve("structure",10,100)
        ledger.settle(rid,10,10,"ok",0.1)
    assert ledger.used() == (6,120)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("structure",1,1)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("audit",20001,1)


def test_failures_retries_cached_response_and_label_count(tmp_path):
    ledger = BudgetLedger(tmp_path/"b.json","a")
    cfg = LLMConfig(backend="openai",max_output_tokens=100,max_pseudo_judgments=3)
    calls = []
    def transport(messages,phase):
        calls.append(phase)
        if len(calls) == 1:
            raise TimeoutError()
        return '{"labels": []}',10,5
    client = BudgetedClient(cfg,ledger,tmp_path/"cache",transport,lambda _:10)
    assert client.ask("labels","hello",judgments=1) == {"labels":[]}
    assert ledger.used() == (2,125)
    assert ledger.data["pseudo_judgments"] == 2
    assert client.ask("labels","hello",judgments=1) == {"labels":[]}
    assert len(calls) == 2
    assert client.ask("labels","new",judgments=2) is None
    assert ledger.data["pseudo_judgments"] == 2


def test_unreliable_tokenizer_never_calls_transport(tmp_path):
    ledger = BudgetLedger(tmp_path/"b.json","a")
    def fail(messages,phase):
        pytest.fail("must not send request")
    client = BudgetedClient(LLMConfig(backend="openai"),ledger,tmp_path,fail)
    assert client.ask("structure","hi") is None
    assert ledger.used() == (0,0)


def test_json_failure_and_accounting_mismatch(tmp_path):
    ledger = BudgetLedger(tmp_path/"b.json","a")
    cfg = LLMConfig(backend="openai",max_output_tokens=20)
    client = BudgetedClient(cfg,ledger,tmp_path/"cache",lambda *_: ("invalid",10,2),lambda _:10)
    assert client.ask("structure","bad") is None
    assert ledger.used() == (2,24)
    client.transport = lambda *_: ('{}',100,100)
    assert client.ask("structure","mismatch") is None
    calls = ledger.used()[0]
    assert client.ask("audit","later") is None
    assert ledger.used()[0] == calls
