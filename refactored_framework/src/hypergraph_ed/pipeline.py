from __future__ import annotations

import json
import platform
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .artifacts import atomic_json, fingerprint, read_json
from .artifacts.lock import run_lock
from .data import read_table, profile_table
from .data.encoding import SemanticStore
from .structure.induce import local_program, induce
from .structure.schema import ConstraintProgram
from .llm import BudgetLedger, BudgetedClient
from .llm.refine import refine
from .models.train import cross_fit
from .operators import execute
from .evidence.schema import Status
from .evidence.localize import localize


def run(input_path, config, output_path, resume=False):
    output = Path(output_path).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with run_lock(output/".run.lock"):
        table = read_table(input_path)
        input_hash = fingerprint(table.to_dict("split"))
        source_root=Path(__file__).parent
        source_id=fingerprint([(str(p.relative_to(source_root)),p.read_text(encoding="utf-8")) for p in sorted(source_root.rglob("*.py"))])
        semantic_files=[]
        if config.model.embedding_backend=="semantic" and config.model.semantic_path and Path(config.model.semantic_path).is_dir():
            root=Path(config.model.semantic_path)
            semantic_files=sorted((str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())
        runtime={"source_fingerprint":source_id,"llm_model":config.llm.model or os.environ.get("LLM_MODEL","") if config.llm.backend=="openai" else config.llm.backend,"llm_endpoint":config.llm.base_url or os.environ.get("LLM_BASE_URL","") if config.llm.backend=="openai" else "","semantic_resources":fingerprint(semantic_files)}
        if config.llm.backend=="fixture":
            runtime["fixture_fingerprint"]=fingerprint(read_json(Path(config.llm.fixture_path)))
        if config.llm.backend=="openai" and config.llm.tokenizer_path and Path(config.llm.tokenizer_path).is_dir():
            tokenizer_root=Path(config.llm.tokenizer_path)
            runtime["tokenizer_fingerprint"]=fingerprint(sorted((str(p.relative_to(tokenizer_root)),p.stat().st_size,p.stat().st_mtime_ns) for p in tokenizer_root.rglob("*") if p.is_file()))
        identity = fingerprint([input_hash,config.model_dump(),__version__,runtime])
        manifest_path = output/"manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity"] != identity:
                raise ValueError("output directory belongs to a different input/configuration")
            if not resume:
                raise ValueError("output already exists; use --resume or a new directory")
            if manifest.get("complete"):
                return read_json(output/"report.json")
        else:
            manifest = {"identity":identity,"input_fingerprint":input_hash,"config":config.model_dump(),"runtime":runtime,"version":__version__,"complete":False,"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__}
            atomic_json(manifest_path,manifest)
        start = time.monotonic()
        timings,issues = {},[]
        # Semantic mode is preflighted before any potentially billable request.
        semantic = SemanticStore(config.model,table,output)
        ledger = BudgetLedger(output/"budget.json",identity)
        client = BudgetedClient(config.llm,ledger,output/"llm_cache")
        stage = time.monotonic()
        program_path = output/"program_initial.json"
        if program_path.exists():
            program = ConstraintProgram.model_validate(read_json(program_path))
        else:
            base = local_program(table)
            program,notes = induce(table,client,base,config.seed)
            issues.extend(notes)
            atomic_json(program_path,program.model_dump())
        atomic_json(output/"profiles.json",profile_table(table))
        timings["structure"] = time.monotonic()-stage
        stage = time.monotonic()
        try:
            scores,residuals,evidence,fold_ids,alternatives = cross_fit(table,program,config,semantic,output/"models_round0")
        except ValueError as exc:
            if "distinct rows" not in str(exc):
                raise
            issues.append(f"local-only degradation: {exc}")
            evidence = execute(table,program)
            scores = np.full(table.shape,0.1,dtype=np.float32)
            for e in evidence:
                if e.status == Status.VIOLATION:
                    scores[e.row_id,table.columns.get_loc(e.column)] = max(scores[e.row_id,table.columns.get_loc(e.column)],e.score*e.reliability)
            residuals,fold_ids,alternatives = np.zeros_like(scores),np.zeros(len(table),dtype=int),{}
        timings["training_round0"] = time.monotonic()-stage
        if config.llm.backend != "disabled" and config.llm.allow_revision and len(set(fold_ids)) > 1:
            stage = time.monotonic()
            revised,pseudo,notes = refine(table,program,scores,evidence,client,config,output/"refinement.json")
            issues.extend(notes)
            if revised.model_dump() != program.model_dump() or pseudo:
                program = revised
                scores,residuals,evidence,fold_ids,alternatives = cross_fit(table,program,config,semantic,output/"models_round1",pseudo)
            timings["refinement_and_training_round1"] = time.monotonic()-stage
        stage = time.monotonic()
        localization = localize(table,program,evidence,fold_ids,alternatives,config)
        timings["localization"] = time.monotonic()-stage
        by_cell = defaultdict(list)
        for e in evidence:
            by_cell[(e.row_id,e.column)].append(e)
        records = []
        for i in range(len(table)):
            for j,col in enumerate(table.columns):
                evs = by_cell[(i,col)]
                active = [e for e in evs if e.status != Status.ABSTAIN]
                positives = [e for e in active if e.status == Status.VIOLATION]
                primary = max(positives,key=lambda e:e.score*e.reliability) if positives else None
                score = float(scores[i,j])
                # Localization is evidence, not an unconditional label override.
                if positives and all(e.details.get("counterevidence") for e in positives):
                    score *= 0.5
                covered_families={e.family for e in active if e.reliability>=0.5}
                uncertain = config.uncertainty_low <= score <= config.uncertainty_high or len(covered_families)<2 or len(set(fold_ids)) < 2
                records.append({"row_id":i,"column":str(col),"value":str(table.iat[i,j]),"is_error":score>=config.error_threshold,"error_score":score,"error_type":primary.error_type if primary else "OTHER" if score>=config.error_threshold else "NONE","low_confidence":uncertain,"suggested_fix":primary.suggested_fix if primary else None,"model_residual":float(residuals[i,j]),"scoring_fold":int(fold_ids[i]),"relations":json.dumps(sorted({e.constraint_id for e in active})),"evidence":json.dumps([e.record() for e in evs],ensure_ascii=False),"localization":json.dumps(localization.get((i,col),[]),ensure_ascii=False)})
        pd.DataFrame(records).to_csv(output/"detections.csv",index=False)
        atomic_json(output/"program.json",program.model_dump())
        issues.extend(client.issues)
        timings["total_this_invocation"] = time.monotonic()-start
        calls,tokens = ledger.used()
        report = {"input_fingerprint":input_hash,"shape":list(table.shape),"constraints":len(program.constraints),"model_backend":config.model.embedding_backend,"semantic":semantic.metadata,"simulated_llm":config.llm.backend=="fixture","llm_calls":calls,"charged_tokens":tokens,"pseudo_judgments":ledger.data["pseudo_judgments"],"cache_hits":ledger.data["cache_hits"],"detected":sum(r["is_error"] for r in records),"low_confidence_fraction":float(np.mean([r["low_confidence"] for r in records])),"timings_seconds":timings,"issues":issues,"cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
        atomic_json(output/"report.json",report)
        manifest["complete"] = True
        atomic_json(manifest_path,manifest)
        return report
