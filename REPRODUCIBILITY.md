# CompatTrajFlow experimental protocol

This document summarizes the configuration and execution order used for the
CompatTrajFlow experiments.

## Data protocol

| Item | Setting |
|---|---|
| Split | chronological 80% train, 10% validation, 10% test |
| Trajectory length | 10–300 matched edges |
| Anomaly share | 5% |
| Anomaly types | Detour, Switch, Time Shift |
| Severity | 0.1 and 0.3 |
| Detour modality factor | `0.01 + 0.99s` |
| Switch | `kappa=3` |
| Time Shift | `delta=30` seconds |

The preprocessing and anomaly-generation commands are described in
[`data/README.md`](data/README.md). All stochastic commands require seeds to be
supplied explicitly. Validation caches and evaluation metadata record the
reconstruction-noise seeds used by those stages.

## Model configuration

| Item | Setting |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Objective | flow vector loss + `0.25` weight-tied token CE |
| DiT depth | 4 |
| Embedding dimension | 128 |
| Hidden dimension | 256 |
| Attention heads | 8 |
| Training-time `t` sampling | logit-normal, `P_mean=-0.8`, `P_std=0.8` |
| Flow denominator epsilon | `0.05` |
| CFG condition-drop probability | `0.1` |
| Per-GPU batch size | 256 |
| EMA decay | 0.9999 |
| Precision | BF16 mixed precision |

Porto is trained for 100 epochs with 10,000 optimizer warm-up steps and five
classifier-free-guidance warm-up epochs. Xi'an is trained for 300 epochs with
3,000 optimizer warm-up steps and fifteen classifier-free-guidance warm-up
epochs.

## Execution order

1. Place the raw data under `data/` and run `prepare_data.py` for both datasets.
2. Train CompatTrajFlow with `scripts/train_porto.sh` and
   `scripts/train_xian.sh`.
3. Run `scripts/run_validation_grid.sh` for each checkpoint.
4. Run `select_fourparam_from_val.py` to freeze the route and temporal
   reconstruction parameters.
5. Run `scripts/evaluate_main_results.sh` with the frozen selection.

The corresponding commands are listed in the top-level [`README.md`](README.md).

## Validation selection

The route branch and temporal branch each select a reconstruction time `t_r`
and classifier-free-guidance scale `w`. Selection uses the equal-weight macro
AUROC over Detour, Switch, and Time Shift validation anomalies at severity
0.3. The two branches are selected independently. Average precision is
reported for analysis but is not used to select parameters.

The supplied validation script evaluates `t_r` from 0.1 to 0.9 in increments
of 0.1 and `w` from 1 to 10. Per-sample scores are averaged over every
user-provided reconstruction-noise seed before parameter selection.

The validation grid stores its scores under:

```text
result/porto_wt_wide_val_grid/
result/xian_wt_wide_val_grid/
```

`select_fourparam_from_val.py` reads these caches and writes the frozen
four-tuple `(t_X, w_X, t_T, w_T)` to:

```text
result/fourparam_zsum/frozen_fourparam_selections.json
```

## Test evaluation

Final evaluation does not search over reconstruction parameters. For each
branch, the evaluator computes the mean and standard deviation of normal
validation scores, standardizes route and temporal scores separately, and
uses:

```text
Z = zX + zT
```

The same validation statistics are applied to normal and anomalous test
trajectories. Test labels are used only to compute the final AUROC and average
precision. They are not used for parameter selection or score calibration.

For each dataset, the evaluator writes:

```text
result/fourparam_zsum/test/<dataset>_wt/test_metrics.csv
result/fourparam_zsum/test/<dataset>_wt/metadata.json
```

`metadata.json` records the checkpoint, frozen selection, user-provided seeds,
calibration statistics, anomaly settings, and score-fusion definition.
