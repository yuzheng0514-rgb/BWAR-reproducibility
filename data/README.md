# Data

The article uses the public 2024 Divvy trip archive and the PEMS-BAY traffic
benchmark. Raw third-party data are not redistributed.

## Divvy

Run

```bash
PYTHONPATH=src python scripts/download_divvy.py
```

to retrieve the twelve official monthly files and verify the SHA-256 values in
`divvy_source_manifest.json`. Source records are governed by the
[Divvy Data License Agreement](https://divvybikes.com/data-license-agreement).

For journal submission, build the processed Supporting Data package with

```bash
PYTHONPATH=src python scripts/prepare_divvy_supporting_data.py \
  --source-dir data/divvy \
  --output-dir /path/to/Divvy_processed_data
```

The resulting ZIP contains only the analysis inputs and provenance needed for
the paper: selected hourly station counts, Gaussian means and covariances,
the complete station-selection audit, chronological splits, metadata, and
checksums. It does not redistribute the original trip-level archives and is
not covered by the repository's MIT software licence.

After extracting the ZIP, reproduce the analysis directly from the processed
counts with

```bash
PYTHONPATH=src python scripts/run_divvy.py \
  --processed-data-dir /path/to/Divvy_processed_data \
  --output-dir results/generated/divvy
```

## PEMS-BAY

Obtain `pems-bay.h5` and `graph_sensor_locations_bay.csv` from the original
PEMS-BAY/DCRNN distribution and place them under `data/pems_bay/`. The analysis
script records the raw HDF5 MD5 digest, selected sensor identifiers, and all
training-defined preprocessing choices in `protocol_lock.json`.

The repository includes compact derived losses and protocol metadata needed to
audit the article, but not stand-alone copies of either raw dataset.
