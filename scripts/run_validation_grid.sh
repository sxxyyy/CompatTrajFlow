#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "usage: $0 <porto|xian> flow_od_wt <checkpoint> <output-dir> <seed> [seed ...]" >&2
  exit 2
fi

if [[ "$2" != "flow_od_wt" ]]; then
  echo "only the CompatTrajFlow module flow_od_wt is supported" >&2
  exit 2
fi

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCATION="$1"
MODEL_MODULE="$2"
CHECKPOINT="$3"
OUTPUT_DIR="$4"
shift 4

"$PYTHON_BIN" validation_grid_search.py \
  --location "$LOCATION" \
  --model_module "$MODEL_MODULE" \
  --checkpoint_path "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --times 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --guidance_scales 1 2 3 4 5 6 7 8 9 10 \
  --seeds "$@" \
  --anomaly_ratio 0.3 \
  --switch_relax 3 \
  --shift_time_gap 30 \
  --batch_size 256 \
  --num_workers 16 \
  --device "${DEVICE:-cuda:0}"
