#!/usr/bin/env python3
"""Build the processed PEMS-BAY Supporting Information package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_pems_bay import (  # noqa: E402
    BLOCK_SIZE,
    CONDITION_Q90_CAP,
    DIMENSION,
    FIT_FRACTION,
    HORIZONS,
    N_PANELS,
    RIDGE_GRID,
    SEED,
    SHRINKAGE_GRID,
    VALIDATION_FRACTION,
    _choose_shrinkage,
    _contiguous_block_starts,
    _load_locations,
    _load_raw,
    _paired_effects,
    _select_spatial_panels,
    _standardize_from_fitting_period,
    _state_moments,
)
from run_pems_bay_from_states import _evaluate_panel, _write_summary  # noqa: E402


REFERENCE_RESULTS = ROOT / "results" / "reference" / "pems_bay"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: int) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


def _standardization_profile(raw: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    fit_rows = int(np.floor(FIT_FRACTION * len(raw)))
    fitting = raw[:fit_rows]
    valid = np.isfinite(fitting) & (fitting > 0.0)
    median = np.nanmedian(np.where(valid, fitting, np.nan), axis=0)
    clean = np.where(np.isfinite(raw) & (raw > 0.0), raw, median[None, :])
    center = clean[:fit_rows].mean(axis=0)
    scale = clean[:fit_rows].std(axis=0, ddof=1)
    standardized = (clean - center) / scale
    profile = pd.DataFrame(
        {
            "source_column_index": np.arange(raw.shape[1]),
            "fitting_valid_fraction": valid.mean(axis=0),
            "fitting_imputation_median_mph": median,
            "fitting_center_mph": center,
            "fitting_scale_mph": scale,
            "fitting_raw_standard_deviation_mph": np.nanstd(
                np.where(valid, fitting, np.nan), axis=0
            ),
        }
    )
    return standardized, profile


def _write_sensor_profile(
    path: Path,
    *,
    panels: list[np.ndarray],
    panel_ids: list[list[int]],
    profile: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel_number, (panel, identifiers) in enumerate(
        zip(panels, panel_ids), start=1
    ):
        for rank, (column, sensor_id) in enumerate(zip(panel, identifiers), start=1):
            values = profile.iloc[int(column)]
            rows.append(
                {
                    "panel": panel_number,
                    "sensor_rank": rank,
                    "sensor_id": int(sensor_id),
                    "fitting_valid_fraction": float(values["fitting_valid_fraction"]),
                    "fitting_imputation_median_mph": float(
                        values["fitting_imputation_median_mph"]
                    ),
                    "fitting_center_mph": float(values["fitting_center_mph"]),
                    "fitting_scale_mph": float(values["fitting_scale_mph"]),
                    "fitting_raw_standard_deviation_mph": float(
                        values["fitting_raw_standard_deviation_mph"]
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False, float_format="%.17g")
    return output


def _write_means(
    path: Path,
    *,
    states: dict[int, tuple[np.ndarray, np.ndarray]],
    panel_ids: list[list[int]],
    starts: np.ndarray,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for panel, (means, _) in states.items():
        sensor_ids = np.asarray(panel_ids[panel - 1], dtype=int)
        for state_index, start in enumerate(starts):
            rows.append(
                pd.DataFrame(
                    {
                        "panel": panel,
                        "state_index": state_index,
                        "block_start": _timestamp(int(timestamps[int(start)])),
                        "block_end_exclusive": _timestamp(
                            int(timestamps[int(start)])
                            + BLOCK_SIZE * 5 * 60 * 1_000_000_000
                        ),
                        "sensor_rank": np.arange(1, DIMENSION + 1),
                        "sensor_id": sensor_ids,
                        "standardized_mean": means[state_index],
                    }
                )
            )
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(path, index=False, float_format="%.17g")
    return output


def _write_covariances(
    path: Path,
    *,
    states: dict[int, tuple[np.ndarray, np.ndarray]],
    panel_ids: list[list[int]],
    starts: np.ndarray,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    row_index, column_index = np.triu_indices(DIMENSION)
    rows: list[pd.DataFrame] = []
    for panel, (_, covariances) in states.items():
        sensor_ids = np.asarray(panel_ids[panel - 1], dtype=int)
        for state_index, start in enumerate(starts):
            rows.append(
                pd.DataFrame(
                    {
                        "panel": panel,
                        "state_index": state_index,
                        "block_start": _timestamp(int(timestamps[int(start)])),
                        "row_sensor_rank": row_index + 1,
                        "column_sensor_rank": column_index + 1,
                        "row_sensor_id": sensor_ids[row_index],
                        "column_sensor_id": sensor_ids[column_index],
                        "standardized_covariance": covariances[
                            state_index, row_index, column_index
                        ],
                    }
                )
            )
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(path, index=False, float_format="%.17g")
    return output


def _write_splits(
    path: Path,
    *,
    starts: np.ndarray,
    timestamps: np.ndarray,
    fit_end: int,
    validation_end: int,
) -> pd.DataFrame:
    partitions = (
        ("fitting", 0, fit_end),
        ("validation", fit_end, validation_end),
        ("testing", validation_end, len(starts)),
    )
    rows = []
    for name, start_state, end_state in partitions:
        first = int(starts[start_state])
        last = int(starts[end_state - 1])
        rows.append(
            {
                "partition": name,
                "start_state_inclusive": start_state,
                "end_state_exclusive": end_state,
                "state_count": end_state - start_state,
                "first_block_start": _timestamp(int(timestamps[first])),
                "last_block_end_exclusive": _timestamp(
                    int(timestamps[last])
                    + BLOCK_SIZE * 5 * 60 * 1_000_000_000
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False)
    return output


def _write_evaluation_origins(
    path: Path,
    *,
    starts: np.ndarray,
    timestamps: np.ndarray,
    validation_end: int,
) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for source in range(validation_end, len(starts) - horizon):
            target = source + horizon
            rows.append(
                {
                    "horizon": horizon,
                    "source_state": source,
                    "target_state": target,
                    "source_block_start": _timestamp(
                        int(timestamps[int(starts[source])])
                    ),
                    "target_block_start": _timestamp(
                        int(timestamps[int(starts[target])])
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False)
    return output


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Processed PEMS-BAY Gaussian states for the BWAR study

## Scope

This Supporting Information package contains the analysis-ready Gaussian
states and result-level files used for the PEMS-BAY application. It excludes
the source `pems-bay.h5` file, the sensor-location file, geographic
coordinates, and all five-minute traffic-speed records.

## Files

- `pems_bay_gaussian_means.csv`: standardized 20-dimensional mean vectors for
  each panel and nonoverlapping 12-hour state.
- `pems_bay_gaussian_covariances.csv`: upper triangles of the corresponding
  shrunk 20-by-20 covariance matrices. Symmetry reconstructs the lower
  triangle.
- `panel_sensor_profile.csv`: panel membership, within-panel order, sensor IDs,
  and fitting-period standardization summaries. No coordinates are included.
- `chronological_splits.csv`: fitting, validation, and testing partitions.
- `evaluation_origins.csv`: source and target states used at each horizon.
- `validation_selected_ridges.csv`: ridge selected for every panel, method,
  and horizon.
- `training_validation_geometry_audit.csv`: geometry diagnostics reported for
  the fitting and validation periods.
- `test_origin_level_wasserstein.csv`: result-level losses underlying the
  PEMS-BAY table; the source-data Gaussian likelihood diagnostic is omitted.
- `test_method_summary_wasserstein.csv`: method-level summaries and
  moving-block intervals for the reported Wasserstein and mean endpoints.
- `test_paired_effects.csv`: paired BWAR-minus-comparator contrasts.
- `table_display.csv`: values used to construct the manuscript table.
- `analysis_protocol.json`: fixed analysis parameters and selected panels.
- `source_provenance.json`: source links, input-file checksums, exclusions, and
  rights notes.
- `verification_report.json`: numerical checks against the archived article
  results.
- `SOURCE_AND_RIGHTS.md`: source and reuse information.
- `processed_data_manifest.json`: file sizes, row counts, and checksums.
- `checksums.sha256`: integrity checks for the package contents.

## Processing

The source dataset contains synchronized five-minute traffic speeds from 325
Bay Area sensors from January through June 2017. Four deterministic geographic
panels of 20 sensors were selected using only fitting-period availability and
sensor locations. Speed values were imputed, centered, and scaled using the
first 60% of source rows. Nonoverlapping blocks of 144 observations were then
summarized by their standardized mean and sample covariance. The covariance
shrinkage value 0.20 was selected from the prespecified grid using only fitting
and validation states. Each panel contains 360 Gaussian states: states 0--215
are the fitting partition, 216--287 are the validation partition, and 288--359
are the testing partition.

## Reproduction

The package is generated by `scripts/prepare_pems_bay_supporting_data.py` in
the BWAR reproducibility repository. The reported Wasserstein analysis can be
rerun without the source HDF5 or location file:

```bash
PYTHONPATH=src python scripts/run_pems_bay_from_states.py \
  --data-dir /path/to/PEMS_BAY_processed_states \
  --out results/generated/pems_bay_from_states
```

The reconstruction script reproduces the total Gaussian Wasserstein loss, its
covariance component, standardized mean RMSE, ridge selections, bootstrap
intervals, and paired contrasts. The unreported source-data Gaussian
likelihood diagnostic requires five-minute measurements and is outside this
processed package.
""",
        encoding="utf-8",
    )


def _write_rights(path: Path) -> None:
    path.write_text(
        """# Source and rights information

The traffic measurements underlying this package originate from the California
Department of Transportation Performance Measurement System (Caltrans PeMS)
and were obtained from the PEMS-BAY distribution provided with the DCRNN
benchmark.

- Caltrans PeMS: https://pems.dot.ca.gov/
- Caltrans PeMS data-source page:
  https://dot.ca.gov/programs/traffic-operations/mpr/pems-source
- Caltrans/PeMS terms of use: https://pems.dot.ca.gov/?view=tou
- DCRNN benchmark: https://github.com/liyaguang/DCRNN
- DCRNN article: Li, Y., Yu, R., Shahabi, C. and Liu, Y. (2018), Diffusion
  Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting,
  International Conference on Learning Representations.

This package does not redistribute `pems-bay.h5`, the DCRNN sensor-location
file, sensor coordinates, or five-minute speed records. It contains only the
study-specific standardized 12-hour Gaussian summaries, analysis metadata,
and outputs underlying the associated manuscript results.

No additional licence is applied to the source-derived data in this package.
The MIT licence in the BWAR code repository applies to the authors' software,
not to the underlying Caltrans or DCRNN data. Users who need the source
measurements or locations should obtain them from the original source and
follow the applicable source terms. No affiliation with or endorsement by
Caltrans, PeMS, or the DCRNN authors is implied.
""",
        encoding="utf-8",
    )


def _reference_wasserstein_files(out: Path) -> None:
    origin = pd.read_csv(REFERENCE_RESULTS / "test_origin_level.csv").drop(
        columns=["raw_gaussian_nll"]
    )
    origin.to_csv(
        out / "test_origin_level_wasserstein.csv",
        index=False,
        float_format="%.17g",
    )
    summary = pd.read_csv(REFERENCE_RESULTS / "test_method_summary.csv").drop(
        columns=["raw_gaussian_nll_mean"]
    )
    summary.to_csv(
        out / "test_method_summary_wasserstein.csv",
        index=False,
        float_format="%.17g",
    )
    for name in (
        "validation_selected_ridges.csv",
        "training_validation_geometry_audit.csv",
        "test_paired_effects.csv",
        "table_display.csv",
    ):
        shutil.copy2(REFERENCE_RESULTS / name, out / name)


def _verify(
    *,
    states: dict[int, tuple[np.ndarray, np.ndarray]],
    panel_ids: list[list[int]],
    shrinkage: float,
    fit_end: int,
    validation_end: int,
    out: Path,
) -> dict[str, object]:
    reference_protocol = json.loads(
        (REFERENCE_RESULTS / "protocol_lock.json").read_text()
    )
    all_records: list[dict[str, float | int | str]] = []
    all_ridges: list[dict[str, float | int | str]] = []
    minimum_eigenvalue = np.inf
    for panel, (means, covariances) in states.items():
        minimum_eigenvalue = min(
            minimum_eigenvalue, float(np.linalg.eigvalsh(covariances).min())
        )
        records, ridges = _evaluate_panel(
            panel, means, covariances, fit_end, validation_end
        )
        all_records.extend(records)
        all_ridges.extend(ridges)

    generated = pd.DataFrame(all_records).sort_values(
        ["panel", "source_state", "horizon", "method"]
    ).reset_index(drop=True)
    reference = pd.read_csv(out / "test_origin_level_wasserstein.csv").sort_values(
        ["panel", "source_state", "horizon", "method"]
    ).reset_index(drop=True)
    keys = ["panel", "source_state", "horizon", "method"]
    numeric = [
        "w2_squared",
        "mean_w2_component",
        "covariance_w2_component",
        "mean_rmse",
    ]
    keys_match = generated[keys].equals(reference[keys])
    maximum_origin_absolute_difference = float(
        np.max(np.abs(generated[numeric].to_numpy() - reference[numeric].to_numpy()))
    )
    origin_values_match = np.allclose(
        generated[numeric], reference[numeric], rtol=1e-10, atol=5e-10
    )

    generated_ridges = pd.DataFrame(all_ridges).sort_values(
        ["panel", "method", "horizon"]
    ).reset_index(drop=True)
    reference_ridges = pd.read_csv(out / "validation_selected_ridges.csv").sort_values(
        ["panel", "method", "horizon"]
    ).reset_index(drop=True)
    ridges_match = generated_ridges.equals(reference_ridges)

    generated_summary_path = out / ".verification_summary.csv"
    generated_summary = _write_summary(generated, generated_summary_path)
    reference_summary = pd.read_csv(out / "test_method_summary_wasserstein.csv")
    generated_summary = generated_summary.sort_values(
        ["method", "horizon"]
    ).reset_index(drop=True)
    reference_summary = reference_summary.sort_values(
        ["method", "horizon"]
    ).reset_index(drop=True)
    summary_keys_match = generated_summary[["method", "horizon"]].equals(
        reference_summary[["method", "horizon"]]
    )
    summary_numeric = [
        column
        for column in generated_summary.columns
        if column in reference_summary.columns
        and column not in ("method", "horizon")
    ]
    summaries_match = np.allclose(
        generated_summary[summary_numeric],
        reference_summary[summary_numeric],
        rtol=1e-11,
        atol=1e-11,
    )
    generated_summary_path.unlink()

    report = {
        "panel_sensor_ids_match_article_reference": panel_ids
        == reference_protocol["panel_sensor_ids"],
        "selected_shrinkage_matches_article_reference": bool(
            np.isclose(
                shrinkage, reference_protocol["selected_covariance_shrinkage"]
            )
        ),
        "state_count_matches_article_reference": all(
            len(means) == reference_protocol["n_states"]
            for means, _ in states.values()
        ),
        "all_covariance_matrices_are_positive_definite": minimum_eigenvalue > 0,
        "minimum_covariance_eigenvalue": minimum_eigenvalue,
        "origin_keys_match_article_reference": keys_match,
        "origin_wasserstein_values_match_article_reference": origin_values_match,
        "maximum_origin_absolute_difference": maximum_origin_absolute_difference,
        "validation_ridges_match_article_reference": ridges_match,
        "summary_keys_match_article_reference": summary_keys_match,
        "summary_wasserstein_values_match_article_reference": summaries_match,
    }
    report["passed"] = all(
        bool(value)
        for key, value in report.items()
        if key
        not in ("minimum_covariance_eigenvalue", "maximum_origin_absolute_difference")
    )
    return report


def _write_protocol(
    path: Path,
    *,
    panel_ids: list[list[int]],
    n_states: int,
    fit_end: int,
    validation_end: int,
    shrinkage: float,
    sensor_audit: dict[str, float],
    condition_audit: dict[str, float],
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "dataset": "PEMS-BAY",
        "seed": SEED,
        "state_construction": "nonoverlapping 12-hour blocks of five-minute readings",
        "block_size": BLOCK_SIZE,
        "dimension": DIMENSION,
        "panel_count": N_PANELS,
        "n_states": n_states,
        "fit_end": fit_end,
        "validation_end": validation_end,
        "horizons": list(HORIZONS),
        "ar_model": "diagonal ridge VAR(1)",
        "ridge_grid": list(RIDGE_GRID),
        "covariance_shrinkage_grid": list(SHRINKAGE_GRID),
        "condition_q90_cap": CONDITION_Q90_CAP,
        "panel_sensor_ids": panel_ids,
        "selected_covariance_shrinkage": shrinkage,
        "sensor_audit": sensor_audit,
        "condition_audit": condition_audit,
        "moving_block_bootstrap": {
            "block_length": 7,
            "draws": 2000,
            "method_level_seed": SEED + 8107,
            "paired_effect_seed": SEED,
        },
    }
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def _write_provenance(
    path: Path,
    *,
    raw_path: Path,
    locations_path: Path,
    timestamps: np.ndarray,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source_dataset": "PEMS-BAY",
        "original_system": "Caltrans Performance Measurement System (PeMS)",
        "distribution_used": "DCRNN benchmark distribution",
        "source_urls": {
            "caltrans_pems": "https://pems.dot.ca.gov/",
            "caltrans_pems_source": "https://dot.ca.gov/programs/traffic-operations/mpr/pems-source",
            "caltrans_pems_terms": "https://pems.dot.ca.gov/?view=tou",
            "dcrnn": "https://github.com/liyaguang/DCRNN",
        },
        "temporal_coverage_as_stored": {
            "first_timestamp": _timestamp(int(timestamps[0])),
            "last_timestamp": _timestamp(int(timestamps[-1])),
            "sampling_interval_minutes": 5,
        },
        "source_input_checksums": {
            "pems-bay.h5": {
                "bytes": raw_path.stat().st_size,
                "md5": md5(raw_path),
                "sha256": sha256(raw_path),
            },
            "graph_sensor_locations_bay.csv": {
                "bytes": locations_path.stat().st_size,
                "md5": md5(locations_path),
                "sha256": sha256(locations_path),
            },
        },
        "not_redistributed": [
            "pems-bay.h5",
            "graph_sensor_locations_bay.csv",
            "five-minute traffic-speed records",
            "sensor coordinates",
        ],
        "included_scope": (
            "study-specific standardized 12-hour Gaussian summaries, analysis "
            "metadata, and result-level outputs"
        ),
        "rights_note": (
            "No additional licence is applied to source-derived data. Source "
            "measurements and locations must be obtained from the original "
            "provider and used under the applicable source terms."
        ),
    }
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def _write_manifest(out: Path, *, protocol: dict[str, object]) -> None:
    row_counts: dict[str, int] = {}
    for file in out.glob("*.csv"):
        row_counts[file.name] = sum(1 for _ in file.open(encoding="utf-8")) - 1
    files = []
    for file in sorted(path for path in out.iterdir() if path.is_file()):
        if file.name in ("processed_data_manifest.json", "checksums.sha256"):
            continue
        files.append(
            {
                "name": file.name,
                "bytes": file.stat().st_size,
                "sha256": sha256(file),
                "rows": row_counts.get(file.name),
            }
        )
    manifest = {
        "dataset_title": (
            "Processed PEMS-BAY Gaussian states for Bures--Wasserstein autoregression"
        ),
        "resource_type": "Supporting data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Gaussian state summaries and result-level evidence; source HDF5, "
            "locations, coordinates, and five-minute speeds are excluded"
        ),
        "processing": protocol,
        "preparation_script": {
            "path": "scripts/prepare_pems_bay_supporting_data.py",
            "sha256": sha256(Path(__file__)),
        },
        "reconstruction_script": {
            "path": "scripts/run_pems_bay_from_states.py",
            "sha256": sha256(ROOT / "scripts" / "run_pems_bay_from_states.py"),
        },
        "files": files,
    }
    manifest_path = out / "processed_data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    checksum_files = sorted(
        path
        for path in out.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (out / "checksums.sha256").write_text(
        "".join(f"{sha256(file)}  {file.name}\n" for file in checksum_files),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    np.random.seed(SEED)

    raw, identifiers, timestamps = _load_raw(args.raw)
    locations = _load_locations(args.locations, identifiers)
    panels, panel_ids, sensor_audit = _select_spatial_panels(
        raw, locations, identifiers
    )
    starts = _contiguous_block_starts(timestamps)
    n_states = len(starts)
    fit_end = int(np.floor(FIT_FRACTION * n_states))
    validation_end = int(np.floor(VALIDATION_FRACTION * n_states))
    standardized, profile = _standardization_profile(raw)
    if not np.allclose(
        standardized,
        _standardize_from_fitting_period(raw),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("standardization profile does not reproduce analysis input")
    shrinkage, condition_audit = _choose_shrinkage(
        standardized, starts, panels, validation_end
    )
    states = {
        panel_number: _state_moments(standardized[:, panel], starts, shrinkage)
        for panel_number, panel in enumerate(panels, start=1)
    }

    _write_sensor_profile(
        args.out / "panel_sensor_profile.csv",
        panels=panels,
        panel_ids=panel_ids,
        profile=profile,
    )
    _write_means(
        args.out / "pems_bay_gaussian_means.csv",
        states=states,
        panel_ids=panel_ids,
        starts=starts,
        timestamps=timestamps,
    )
    _write_covariances(
        args.out / "pems_bay_gaussian_covariances.csv",
        states=states,
        panel_ids=panel_ids,
        starts=starts,
        timestamps=timestamps,
    )
    _write_splits(
        args.out / "chronological_splits.csv",
        starts=starts,
        timestamps=timestamps,
        fit_end=fit_end,
        validation_end=validation_end,
    )
    _write_evaluation_origins(
        args.out / "evaluation_origins.csv",
        starts=starts,
        timestamps=timestamps,
        validation_end=validation_end,
    )
    protocol = _write_protocol(
        args.out / "analysis_protocol.json",
        panel_ids=panel_ids,
        n_states=n_states,
        fit_end=fit_end,
        validation_end=validation_end,
        shrinkage=shrinkage,
        sensor_audit=sensor_audit,
        condition_audit=condition_audit,
    )
    _write_provenance(
        args.out / "source_provenance.json",
        raw_path=args.raw,
        locations_path=args.locations,
        timestamps=timestamps,
    )
    _reference_wasserstein_files(args.out)
    _write_readme(args.out / "README.md")
    _write_rights(args.out / "SOURCE_AND_RIGHTS.md")

    report = _verify(
        states=states,
        panel_ids=panel_ids,
        shrinkage=shrinkage,
        fit_end=fit_end,
        validation_end=validation_end,
        out=args.out,
    )
    (args.out / "verification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError("processed package failed reference-result verification")
    _write_manifest(args.out, protocol=protocol)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_states_per_panel": n_states,
                "panels": len(states),
                "verification_passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
