from __future__ import annotations

import pandas as pd

from ..data.table import is_blank, read_table


def evaluate(dirty_path, clean_path, prediction_path) -> dict:
    dirty, clean = read_table(dirty_path), read_table(clean_path)
    if list(dirty.columns) != list(clean.columns) or dirty.shape != clean.shape:
        raise ValueError("dirty/clean schema and shape must match")
    norm = lambda v: "" if is_blank(v) else str(v)
    truth = {(i,c) for c in dirty.columns for i,(a,b) in enumerate(zip(dirty[c],clean[c])) if norm(a) != norm(b)}
    pred = pd.read_csv(prediction_path, keep_default_na=False)
    if pred.duplicated(["row_id","column"]).any():
        raise ValueError("duplicate prediction cells")
    if len(pred) != dirty.size or not set(pred["column"]) <= set(dirty.columns) or not pred.row_id.between(0,len(dirty)-1).all():
        raise ValueError("predictions must cover every cell exactly once")
    def boolean(v):
        if str(v).lower() not in {"true","false","1","0"}:
            raise ValueError("invalid boolean prediction")
        return str(v).lower() in {"true","1"}
    predicted = {(int(r.row_id),r.column) for r in pred.itertuples() if boolean(r.is_error)}
    def metrics(a,b):
        tp,fp,fn = len(a&b),len(a-b),len(b-a)
        precision,recall = tp/max(1,tp+fp),tp/max(1,tp+fn)
        return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*tp/max(1,2*tp+fp+fn)}
    return {"overall": metrics(predicted,truth), "by_column": {c:metrics({x for x in predicted if x[1]==c},{x for x in truth if x[1]==c}) for c in dirty.columns}, "low_confidence_fraction": float(pred.low_confidence.map(boolean).mean()) if "low_confidence" in pred else None}
