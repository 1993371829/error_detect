from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..artifacts import fingerprint, atomic_json
from .table import is_blank, parse_number, profile_table


def lexical(value: str) -> np.ndarray:
    out = np.zeros(48, dtype=np.float32)
    n = max(1, len(value))
    out[1:8] = [float(is_blank(value)), min(len(value) / 100, 5), sum(c.isdigit() for c in value) / n, sum(c.isalpha() for c in value) / n, sum(c.isspace() for c in value) / n, sum(not c.isalnum() and not c.isspace() for c in value) / n, float(value.isupper())]
    for size in (1, 2, 3):
        for i in range(max(1, len(value) - size + 1)):
            digest = hashlib.blake2b(value[i:i+size].encode("utf-8"), digest_size=4).digest()
            out[16 + int.from_bytes(digest, "little") % 32] += 1
    norm = np.linalg.norm(out[16:])
    if norm:
        out[16:] /= norm
    return out


class SemanticStore:
    def __init__(self, config, table, output: Path):
        self.values = {}
        self.dim = 0
        self.metadata = {"backend": config.embedding_backend}
        if config.embedding_backend == "lexical":
            return
        model_path = Path(config.semantic_path)
        if not config.semantic_path or not model_path.is_dir():
            raise RuntimeError("semantic model missing; run 'hypergraph-ed prepare-model' on the server and set SEMANTIC_MODEL_PATH")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the [semantic] extra; no implicit lexical fallback") from exc
        files = sorted((str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns) for p in model_path.rglob("*") if p.is_file())
        model_id = fingerprint(files)
        unique = sorted(set(str(v) for v in table.to_numpy().ravel()))
        cache_id = fingerprint([model_id, unique])
        cache = output / "embeddings" / f"{cache_id}.npz"
        if cache.exists():
            matrix = np.load(cache, allow_pickle=False)["vectors"]
        else:
            model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False)
            matrix = model.encode(unique, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, vectors=matrix)
            del model
        self.dim = matrix.shape[1]
        self.values = dict(zip(unique, matrix))
        self.metadata.update(model_fingerprint=model_id, vector_dim=self.dim, cache_id=cache_id)
        atomic_json(output / "embeddings" / "manifest.json", self.metadata)

    def vector(self, value: str) -> np.ndarray:
        # Synthetic unseen perturbations retain lexical features; not API encoded.
        return self.values.get(value, np.zeros(self.dim, dtype=np.float32))


class CellEncoder:
    def __init__(self, reference, semantic: SemanticStore, max_categories: int):
        self.columns = list(reference.columns)
        self.semantic = semantic
        self.dim = 48 + semantic.dim
        self.specs = []
        self.category_count = 1
        for p in profile_table(reference):
            col = p["name"]
            counts = reference[col].value_counts()
            parsed = [x[0] for x in map(parse_number, reference[col]) if x]
            median = float(np.median(parsed)) if parsed else 0.
            scale = float(np.subtract(*np.percentile(parsed, [75, 25]))) if parsed else 1.
            scale = max(scale, 1e-3)
            cats = sorted(counts.index) if p["kind"] != "numeric" and len(counts) <= max_categories else []
            vocab = {str(v): i+1 for i, v in enumerate(cats)}
            kind = "numeric" if p["kind"] == "numeric" else "categorical" if cats else "text"
            self.specs.append({"name": col, "kind": kind, "median": median, "scale": scale, "vocab": vocab, "offset": self.category_count, "frequency": {str(k): int(v)/len(reference) for k, v in counts.items()}, "size": len(vocab)+1 if kind == "categorical" else 1 if kind == "numeric" else self.dim})
            self.category_count += len(vocab) + 1

    def value(self, column: int, value: str) -> tuple[np.ndarray, int, int]:
        spec = self.specs[column]
        features = lexical(value)
        number = parse_number(value)
        if number:
            features[0] = np.clip((number[0] - spec["median"]) / spec["scale"], -20, 20)
            features[9] = bool(number[1])
        features[8] = spec["frequency"].get(value, 0.)
        if self.semantic.dim:
            features = np.concatenate([features, self.semantic.vector(value)])
        label = spec["vocab"].get(value, 0)
        return features, spec["offset"] + label if spec["kind"] == "categorical" else 0, label

    def transform(self, table):
        features = np.zeros((len(table), len(self.columns), self.dim), dtype=np.float32)
        category_ids = np.zeros(features.shape[:2], dtype=np.int64)
        targets = np.zeros(features.shape[:2], dtype=np.int64)
        for i, row in enumerate(table.itertuples(index=False, name=None)):
            for j, value in enumerate(row):
                features[i,j], category_ids[i,j], targets[i,j] = self.value(j, str(value))
        return features, category_ids, targets
