# Data preparation

The raw and processed datasets are stored locally under `data/` and are not
included in this repository.

## Data sources

- Porto taxi trajectories: [Taxi Trajectory dataset on Kaggle](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)
- DiDi trajectory data: [DiDi Open Data](https://outreach.didichuxing.com/research/opendata/en/)

The code uses `xian` as the dataset identifier for the DiDi split used by the
experiments.

## Raw file layout

Place the downloaded files in the following structure:

```text
data/
  porto/
    raw/
      train.csv
      graph.graphml              # optional
  xian/
    raw/
      2016_xian/
        2016_10_01.csv
        ...
        2016_10_31.csv
      graph.graphml              # optional
```

If `graph.graphml` is absent, the preprocessing pipeline obtains the driving
road network through OSMnx.

## FMM map matching

Build the supplied FMM Docker image from the repository root:

```bash
docker build -f utils/Dockerfile.fmm -t fmm:0.1.0 .
```

During preprocessing, Python mounts `data/` into the container, runs `stmatch`,
and removes the container after completion. To use another local image tag,
set `FMM_DOCKER_IMAGE` before running `prepare_data.py`.

## Preprocessing commands

Choose an integer seed and pass it explicitly through `$SEED`:

```bash
python prepare_data.py --location porto --seed "$SEED"
python prepare_data.py --location xian --seed "$SEED"
```

For each dataset, the command:

1. formats the raw trajectories;
2. map-matches GPS points to road-network edges;
3. keeps trajectories containing 10–300 matched edges;
4. creates a chronological 80%/10%/10% train/validation/test split;
5. generates Detour, Switch, and Time Shift anomalies for validation and test.

The paper settings use anomaly severities 0.1 and 0.3, a 5% anomaly share,
Switch `kappa=3`, and Time Shift `delta=30` seconds. The Detour modality factor
is `0.01 + 0.99s`, where `s` is the modality score used during candidate
selection.

Processed trajectories and generated anomalies are written below:

```text
data/<dataset>/processed/
data/<dataset>/anomaly/
```
