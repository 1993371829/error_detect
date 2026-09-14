"""Six-table/five-seed runner. Each run has its own budget; no paid defaults."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--datasets",default="hospital,flights,beers,rayyan,movies,billionaire")
    parser.add_argument("--seeds",default="0,1,2,3,4")
    parser.add_argument("--sample-rows",type=int,default=0,help="0=full dataset; positive=deterministic compatibility smoke subset")
    parser.add_argument("--evaluate",action="store_true",help="Clean data is accessed only after detection, in a separate process")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summaries = []
    for dataset in args.datasets.split(","):
        dirty = args.data_dir/f"{dataset}_dirty.csv"
        clean = args.data_dir/f"{dataset}_clean.csv"
        for seed in map(int,args.seeds.split(",")):
            out = args.output/dataset/f"seed_{seed}"
            selected = dirty
            evaluation_clean = clean
            if args.sample_rows:
                sample_dir = args.output/"samples"/dataset
                sample_dir.mkdir(parents=True,exist_ok=True)
                selected = sample_dir/"dirty.csv"
                pd.read_csv(dirty,dtype=str,keep_default_na=False).head(args.sample_rows).to_csv(selected,index=False)
            cmd = [sys.executable,"-B","-m","hypergraph_ed","run","--input",str(selected),"--config",str(args.config.resolve()),"--output",str(out),"--seed",str(seed),"--resume"]
            with (args.output/f"{dataset}_{seed}.log").open("w",encoding="utf-8") as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
            record = json.loads((out/"report.json").read_text(encoding="utf-8"))
            record.update(dataset=dataset,seed=seed)
            if args.evaluate:
                if args.sample_rows:
                    evaluation_clean = selected.parent/"clean.csv"
                    pd.read_csv(clean,dtype=str,keep_default_na=False).head(args.sample_rows).to_csv(evaluation_clean,index=False)
                subprocess.run([sys.executable,"-B","-m","hypergraph_ed","evaluate","--dirty",str(selected),"--clean",str(evaluation_clean),"--predictions",str(out/"detections.csv"),"--output",str(out/"evaluation.json")],check=True,stdout=subprocess.DEVNULL)
                record["metrics"] = json.loads((out/"evaluation.json").read_text(encoding="utf-8"))["overall"]
            summaries.append(record)
            (args.output/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"{dataset} seed={seed}: {record['shape']}, {record['llm_calls']} calls, {record['charged_tokens']} tokens",flush=True)
    if args.evaluate:
        # Bootstrap datasets, preserving paired seeds within each dataset.
        datasets = sorted({r['dataset'] for r in summaries})
        averages = np.array([np.mean([r['metrics']['f1'] for r in summaries if r['dataset']==d]) for d in datasets])
        rng = np.random.default_rng(42)
        draws = [np.mean(rng.choice(averages,len(averages),replace=True)) for _ in range(2000)]
        result = {"macro_f1":float(averages.mean()),"dataset_bootstrap_95ci":np.percentile(draws,[2.5,97.5]).tolist(),"note":"Descriptive interval; not a paired comparison with the legacy baseline."}
        (args.output/"aggregate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
