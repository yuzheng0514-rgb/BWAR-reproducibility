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

## PEMS-BAY

Obtain `pems-bay.h5` and `graph_sensor_locations_bay.csv` from the original
PEMS-BAY/DCRNN distribution and place them under `data/pems_bay/`. The analysis
script records the raw HDF5 MD5 digest, selected sensor identifiers, and all
training-defined preprocessing choices in `protocol_lock.json`.

The repository includes compact derived losses and protocol metadata needed to
audit the article, but not stand-alone copies of either raw dataset.
