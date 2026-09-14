#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "Usage: run_server.sh DIRTY_CSV OUTPUT_DIRECTORY [--resume]" >&2
  exit 2
fi
python -m hypergraph_ed doctor --config "$project_dir/configs/server.yaml"
python -m hypergraph_ed run --input "$1" --output "$2" --config "$project_dir/configs/server.yaml" "${@:3}"
