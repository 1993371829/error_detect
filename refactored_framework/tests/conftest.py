from pathlib import Path
import socket

import pytest

from hypergraph_ed.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must not access network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def config():
    return load_config(ROOT / "configs/local_smoke.yaml")


@pytest.fixture
def fixture_dir():
    return ROOT / "tests/fixtures"
