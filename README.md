# BWAR

Code and compact result artifacts for **Bures--Wasserstein Autoregression for
Gaussian Distributional Time Series** by Yuzheng Dong, Junlie Huang, Chengshuo Du, and Cheng
Meng.

BWAR encodes Gaussian states with affine maps induced by Bures--Wasserstein
geometry, fits a centered ridge lag-design VAR in the resulting coordinates,
and reconstructs Gaussian forecasts through the inverse chart. The fixed
training-barycenter construction is the base method; the rolling-reference
construction is an extension evaluated separately.

## Repository contents

- `src/bwar/`: Gaussian geometry, fixed BWAR, local BWAR, and experiment code.
- `scripts/run_simulations.py`: 80-replication fixed-reference geometry study (S1).
- `scripts/run_continuous_covariance_regimes.py`: 60-replication continuous-covariance reference study (S2).
- `scripts/run_divvy.py`: chronological Divvy target-level analysis.
- `scripts/run_pems_bay.py`: chronological PEMS-BAY panel analysis.
- `scripts/build_artifacts.py`: manuscript Tables 1--3 and Figure 3 builder.
- `results/reference/`: compact results corresponding to the submitted article.
- `artifacts/reference/`: manuscript-ready tables and the S2 figure.

Each experiment contains exactly the methods reported for that experiment.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Rebuild article artifacts from compact results

```bash
make figures
```

Outputs are written to `artifacts/generated/`. The generated tables are checked
against the frozen files in `artifacts/reference/` by the test suite.

## Rerun the simulations

```bash
make s1  # 80 replications, diagonal ridge VAR(1)
make s2  # 60 paired replications, two continuous covariance regimes
```

Use `make smoke` for small implementation checks. Full simulations can be
computationally intensive.

S2 compares a smoothly displaced covariance center with recurrent covariance
and mean cycles around a fixed center. Fixed and local BWAR use the same
trailing window and are tuned chronologically; they differ in whether the
fitting-block Bures reference is retained or recomputed causally.

## Divvy application

```bash
PYTHONPATH=src python scripts/download_divvy.py
make divvy
```

The download script retrieves the official 2024 Divvy trip archives and
verifies their checksums. Raw trip data are not redistributed. The analysis
uses horizons 3, 4, and 5, five chronological origins, 183 common targets per
horizon, and an origin-preserving moving-block bootstrap with 10,000 resamples.

## PEMS-BAY application

Place `pems-bay.h5` and `graph_sensor_locations_bay.csv` under
`data/pems_bay/`, then run:

```bash
make pems
```

PEMS-BAY is not redistributed. The script uses four training-defined
20-sensor geographic panels, nonoverlapping 12-hour Gaussian states, a
chronological fit--validation--test split, and moving-block uncertainty.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the article-item map and exact
protocol details.

## License

The authors' code is released under the MIT License. Divvy data remain subject
to the [Divvy Data License Agreement](https://divvybikes.com/data-license-agreement).
Third-party PEMS-BAY data remain subject to their original distribution terms.
Citation metadata are provided in `CITATION.cff`.
