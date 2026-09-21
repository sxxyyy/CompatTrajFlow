# CompatTrajFlow

This repository contains the code accompanying the submission **CompatTrajFlow:
Modeling Conditional Route–Time Compatibility for Trajectory Anomaly
Detection**.

CompatTrajFlow decomposes trajectory compatibility into two conditional
relations. The route branch models `p(X | O, D)`, while the temporal branch
models `p(T | X)`. Both branches are implemented with conditional flow
matching and DiT backbones. The route branch measures whether a path is
compatible with its origin and destination; the temporal branch measures
whether the observed timing is compatible with that path.

Each branch predicts clean trajectory embeddings from noisy inputs. Its output
is compared with the corresponding input embedding table through a
weight-tied cosine head. At evaluation time, the route and temporal
reconstruction scores are standardized using normal validation trajectories
and combined as `Z = zX + zT`.

The repository provides:

- CompatTrajFlow training, validation, and evaluation code;
- trajectory preprocessing and anomaly generation;
- the baseline implementations used by the same experimental pipeline.

## Repository structure

```text
compattrajflow.py               CompatTrajFlow entry point
flow_od_wt.py                   model, data module, training, and scoring
dit.py                          DiT backbone
dit_od.py                       origin–destination-conditioned DiT
prepare_data.py                 preprocessing and anomaly generation
validation_grid_search.py       validation-only score grid
select_fourparam_from_val.py    validation-only parameter selection
evaluate_fourparam_zsum.py      frozen-parameter evaluation
model/ and runner/              baseline models and runners
main.py / run_experiments.py    baseline entry points
scripts/                        training and evaluation commands
```

## Environment

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate compattrajflow
```

The environment targets Python 3.12. Install a PyTorch build compatible with
the CUDA driver on the target system.

Preprocessing uses Fast Map Matching (FMM) through Docker. Build the supplied
image once before preparing the data:

```bash
docker build -f utils/Dockerfile.fmm -t fmm:0.1.0 .
```

The Python preprocessing pipeline starts the container, runs `stmatch`, and
removes the container when map matching finishes. Set `FMM_DOCKER_IMAGE` if a
different local image tag should be used.

Weights & Biases is disabled by default. It can be enabled with
`WANDB_MODE=offline` or `WANDB_MODE=online`.

## Seed input

Every stochastic entry point requires an explicit seed. The commands below use
`$SEED` to denote an integer chosen by the user. For repeated runs, use
separately chosen variables such as `$SEED_1` and `$SEED_2`.

## Data preparation

Arrange the raw files as described in [`data/README.md`](data/README.md), then
run:

```bash
python prepare_data.py --location porto --seed "$SEED"
python prepare_data.py --location xian --seed "$SEED"
```

The pipeline performs map matching, filters trajectories by length, creates a
chronological 8:1:1 split, and generates Detour, Switch, and Time Shift
validation/test anomalies at severities 0.1 and 0.3.

## Training CompatTrajFlow

The paper configuration uses two GPUs and a per-GPU batch size of 256:

```bash
bash scripts/train_porto.sh "$SEED"
bash scripts/train_xian.sh "$SEED"
```

To run on one GPU:

```bash
GPUS=1 bash scripts/train_porto.sh "$SEED"
GPUS=1 bash scripts/train_xian.sh "$SEED"
```

Checkpoints are written under `checkpoints/`. The default final checkpoint
paths used by the evaluation scripts are:

```text
checkpoints/porto_compattrajflow_last.ckpt
checkpoints/xian_compattrajflow_last.ckpt
```

## Validation-only parameter selection

First score the validation split for each trained checkpoint:

```bash
bash scripts/run_validation_grid.sh \
  porto flow_od_wt checkpoints/porto_compattrajflow_last.ckpt \
  result/porto_wt_wide_val_grid "$SEED"

bash scripts/run_validation_grid.sh \
  xian flow_od_wt checkpoints/xian_compattrajflow_last.ckpt \
  result/xian_wt_wide_val_grid "$SEED"
```

More than one reconstruction-noise seed may be supplied to each command. The
script averages per-sample scores over the supplied seeds.

Freeze the route and temporal parameters using validation data only:

```bash
python select_fourparam_from_val.py
```

The selected parameters are written to:

```text
result/fourparam_zsum/frozen_fourparam_selections.json
```

## Final evaluation

Run final evaluation with one explicitly supplied reconstruction-noise seed:

```bash
bash scripts/evaluate_main_results.sh "$SEED"
```

The evaluator loads the frozen validation selection, estimates branch
standardization statistics from normal validation trajectories, and evaluates
the test split without changing the selected parameters. Each dataset produces
`test_metrics.csv` and `metadata.json` under
`result/fourparam_zsum/test/`.

The complete experimental sequence and configuration are summarized in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Baselines

The included baselines use the same processed trajectories, anomaly sets, and
evaluation interface. Repeated baseline runs can be launched with explicit
seeds:

```bash
python run_experiments.py \
  --location porto \
  --seeds "$SEED_1" "$SEED_2" \
  --models deep_tea causal_tad gmvsae vsae mst_oatd fotraj
```

iBAT has a dedicated entry point:

```bash
python scripts/run_ibat.py --random_seed "$SEED"
```

Traj-MLLM runs through the common baseline entry point and reads its credential
from the `OPENAI_API_KEY` environment variable.
