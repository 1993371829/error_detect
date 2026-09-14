from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


MISSING = {"", "empty", "null", "n/a", "na", "none", "-"}


def is_blank(value: object) -> bool:
    return value is None or str(value).strip().lower() in MISSING


def parse_number(value: object) -> tuple[float, str] | None:
    text = str(value).strip().lower()
    if "," in text:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*[a-z%]*\.?",text):
            return None
        text = text.replace(",", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z%]*)\.?", text)
    if not match:
        return None
    number = float(match.group(1))
    if not np.isfinite(number):
        return None
    units = {"mins": "minute", "min": "minute", "minutes": "minute", "hrs": "hour", "hr": "hour", "hours": "hour", "%": "percent", "oz": "ounce", "ounces": "ounce"}
    unit = units.get(match.group(2), match.group(2))
    return number, unit


def pattern(value: object) -> str:
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", str(value))))


def read_table(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header or len(set(header)) != len(header) or any(not x for x in header):
        raise ValueError("CSV needs unique, non-empty column names")
    table = pd.read_csv(path, dtype=str, keep_default_na=False).reset_index(drop=True)
    if table.empty:
        raise ValueError("empty tables cannot be trained")
    return table


def semantic_type(name: str) -> str:
    # Token boundaries prevent duration->ratio and average->age collisions.
    words = set(re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().replace("_", " ").split())
    for kind, names in [("duration", {"duration", "runtime"}), ("age", {"age"}), ("percentage", {"percent", "percentage", "ratio", "rate"}), ("year", {"year"}), ("latitude", {"latitude"}), ("longitude", {"longitude"})]:
        if words & names:
            return kind
    return "unknown"


def profile_table(table: pd.DataFrame) -> list[dict]:
    result = []
    for col in table.columns:
        s = table[col].astype(str)
        nonblank = s[~s.map(is_blank)]
        parsed = [p for p in map(parse_number, nonblank) if p is not None]
        numeric = bool(len(nonblank)) and len(parsed) / len(nonblank) >= 0.8
        vals = np.array([p[0] for p in parsed])
        counts = nonblank.value_counts()
        units = sorted({p[1] for p in parsed})
        result.append({"name": str(col), "kind": "numeric" if numeric else "categorical" if len(counts) <= 500 else "text", "semantic_type": semantic_type(str(col)), "units": units, "missing_ratio": float(s.map(is_blank).mean()), "distinctness": len(counts) / max(1, len(nonblank)), "top_values": [[v, int(n)] for v, n in counts.head(8).items()], "rare_values": list(counts.tail(4).index), "patterns": list(nonblank.map(pattern).value_counts().head(5).index), "summary": {"min": float(vals.min()), "max": float(vals.max()), "median": float(np.median(vals))} if len(vals) else {}})
    return result


def row_keys(table: pd.DataFrame) -> list[str]:
    return [hashlib.sha256(str(tuple(row)).encode("utf-8")).hexdigest() for row in table.itertuples(index=False, name=None)]


def assign_folds(table: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    keys = row_keys(table)
    unique = sorted(set(keys))
    if len(unique) < count:
        raise ValueError(f"cross-fitting requires at least {count} distinct rows")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {key: i % count for i, key in enumerate(unique)}
    return np.array([mapping[x] for x in keys], dtype=np.int64)
