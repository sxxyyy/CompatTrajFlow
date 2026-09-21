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
GPUS="${GPUS:-2}"
STRATEGY="${STRATEGY:-auto}"
if [[ "$GPUS" -gt 1 && "$STRATEGY" == "auto" ]]; then
  STRATEGY="ddp"
fi

"$PYTHON_BIN" compattrajflow.py \
  --seed "$SEED" \
  --location porto \
  --status train \
  --run_name porto_compattrajflow \
  --gpus "$GPUS" \
  --strategy "$STRATEGY" \
  --batch_size 256 \
  --epochs 100 \
  --depth 4 \
  --emb_dim 128 \
  --hidden_dim 256 \
  --nheads 8 \
  --learning_rate 1e-4 \
  --warmup_steps 10000 \
  --ema_decay 0.9999 \
  --P_mean -0.8 \
  --P_std 0.8 \
  --t_eps 0.05 \
  --cfg_drop_prob 0.1 \
  --cfg_warmup_epochs 5 \
  --precision bf16-mixed \
  --wandb_mode "${WANDB_MODE:-disabled}" \
  --metrics_out result/training/porto_metrics.json
