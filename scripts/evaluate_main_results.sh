#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <seed>" >&2
  exit 2
fi
SEED="$1"

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

SELECTION="${SELECTION:-result/fourparam_zsum/frozen_fourparam_selections.json}"
PORTO_CKPT="${PORTO_CKPT:-checkpoints/porto_compattrajflow_last.ckpt}"
XIAN_CKPT="${XIAN_CKPT:-checkpoints/xian_compattrajflow_last.ckpt}"

"$PYTHON_BIN" evaluate_fourparam_zsum.py \
  --split test --dataset porto --head wt --model_module flow_od_wt \
  --checkpoint_path "$PORTO_CKPT" --selection_json "$SELECTION" \
  --output_dir result/fourparam_zsum/test/porto_wt \
  --seeds "$SEED" \
  --device "${PORTO_DEVICE:-cuda:0}"

"$PYTHON_BIN" evaluate_fourparam_zsum.py \
  --split test --dataset xian --head wt --model_module flow_od_wt \
  --checkpoint_path "$XIAN_CKPT" --selection_json "$SELECTION" \
  --output_dir result/fourparam_zsum/test/xian_wt \
  --seeds "$SEED" \
  --device "${XIAN_DEVICE:-cuda:0}"
