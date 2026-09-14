from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from .config import load_config


def doctor(config):
    import torch
    checks = {"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"requested_device":config.model.device,"embedding_backend":config.model.embedding_backend,"llm_backend":config.llm.backend,"paid_requests":0}
    problems = []
    if config.model.device.startswith("cuda") and not torch.cuda.is_available():
        problems.append("CUDA unavailable")
    if config.model.embedding_backend == "semantic":
        if importlib.util.find_spec("sentence_transformers") is None:
            problems.append("install semantic extra: pip install -e '.[semantic]'")
        if not config.model.semantic_path or not Path(config.model.semantic_path).is_dir():
            problems.append("set SEMANTIC_MODEL_PATH to a prepared local BAAI/bge-m3 directory")
    if config.llm.backend == "openai":
        if not os.environ.get("LLM_API_KEY"):
            problems.append("LLM_API_KEY missing")
        if not (config.llm.model or os.environ.get("LLM_MODEL")):
            problems.append("LLM_MODEL missing")
        if not config.llm.tokenizer_path or not Path(config.llm.tokenizer_path).is_dir():
            problems.append("set LLM_TOKENIZER_PATH to a local tokenizer matching the service")
        if importlib.util.find_spec("transformers") is None:
            problems.append("transformers unavailable for token counting")
    checks.update(ok=not problems,problems=problems)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hypergraph-ed")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--config",required=True)
    r = sub.add_parser("run")
    r.add_argument("--input",required=True)
    r.add_argument("--config",required=True)
    r.add_argument("--output",required=True)
    r.add_argument("--resume",action="store_true")
    r.add_argument("--seed",type=int)
    e = sub.add_parser("evaluate")
    e.add_argument("--dirty",required=True)
    e.add_argument("--clean",required=True)
    e.add_argument("--predictions",required=True)
    e.add_argument("--output")
    prepare = sub.add_parser("prepare-model")
    prepare.add_argument("--model-id",default="BAAI/bge-m3")
    prepare.add_argument("--destination",required=True)
    prepare.add_argument("--revision",default="main")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor(load_config(args.config))
        print(json.dumps(result,ensure_ascii=True,indent=2))
        return 0 if result["ok"] else 1
    if args.command == "prepare-model":
        from huggingface_hub import snapshot_download, HfApi
        revision = HfApi().model_info(args.model_id,revision=args.revision).sha
        location = snapshot_download(repo_id=args.model_id,revision=revision,local_dir=args.destination)
        from .artifacts import atomic_json
        atomic_json(Path(location)/"prepared_revision.json",{"model_id":args.model_id,"revision":revision})
        print(json.dumps({"path":location,"revision":revision}))
        return 0
    if args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.dirty,args.clean,args.predictions)
        if args.output:
            from .artifacts import atomic_json
            atomic_json(Path(args.output),result)
    else:
        from .pipeline import run
        config = load_config(args.config)
        if args.seed is not None:
            config.seed = args.seed
        result = run(args.input,config,args.output,resume=args.resume)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return 0
