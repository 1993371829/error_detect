from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str = "relation_attention"
    hidden_dim: int = 256
    heads: int = 4
    layers: int = 2
    epochs: int = 200
    patience: int = 15
    learning_rate: float = 0.001
    batch_size: int = 128
    folds: int = 3
    max_train_cells: int = 20000
    max_relations: int = 6
    max_peers: int = 8
    device: str = "auto"
    threads: int = 2
    embedding_backend: Literal["lexical", "semantic"] = "lexical"
    semantic_path: str = ""
    max_categories: int = 500
    input_dim: int = 48
    column_count: int = 1
    category_count: int = 1
    target_sizes: list[int] = Field(default_factory=list)
    target_kinds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_dimensions(self):
        for name in ("hidden_dim", "heads", "layers", "epochs", "patience", "batch_size", "max_train_cells", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.heads or self.folds < 2:
            raise ValueError("hidden_dim must divide heads; folds must be >= 2")
        if not 1 <= self.max_relations <= 6 or not 1 <= self.max_peers <= 8:
            raise ValueError("at most six relations and eight peers are supported")
        return self


class LLMConfig(StrictModel):
    backend: Literal["disabled", "fixture", "openai"] = "disabled"
    fixture_path: str = ""
    model: str = ""
    base_url: str = ""
    tokenizer_path: str = ""
    token_counter: Literal["fixture_bytes", "transformers"] = "transformers"
    # Added to the chat template count before reserving a request.
    framing_margin: int = 64
    max_output_tokens: int = 4096
    timeout: float = 120
    temperature: float = 0
    max_pseudo_judgments: int = 128
    allow_revision: bool = True


class Config(StrictModel):
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    error_threshold: float = 0.5
    uncertainty_low: float = 0.35
    uncertainty_high: float = 0.65
    min_peer_support: int = 3
    max_counterfactual_candidates: int = 3
    max_counterfactual_rows: int = 20000

    @model_validator(mode="after")
    def validate_limits(self):
        if not 0 <= self.uncertainty_low <= self.error_threshold <= self.uncertainty_high <= 1:
            raise ValueError("invalid detection thresholds")
        if not 0 <= self.llm.max_pseudo_judgments <= 128:
            raise ValueError("pseudo judgments must be <=128")
        if self.min_peer_support < 2 or not 1 <= self.max_counterfactual_candidates <= 3:
            raise ValueError("invalid localization limits")
        if self.max_counterfactual_rows < 1:
            raise ValueError("counterfactual row cap must be positive")
        return self


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    def read(p: Path, seen: set[Path]) -> dict:
        if p in seen:
            raise ValueError("configuration extends cycle")
        value = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parent = value.pop("extends", None)
        return _merge(read((p.parent / parent).resolve(), seen | {p}), value) if parent else value
    cfg = Config.model_validate(read(path, set()))
    for obj, name in [(cfg.llm, "fixture_path"), (cfg.llm, "tokenizer_path"), (cfg.model, "semantic_path")]:
        value = os.path.expandvars(getattr(obj, name))
        if value and "$" not in value:
            setattr(obj, name, str((path.parent / value).resolve()) if not Path(value).is_absolute() else value)
    return cfg
