#!/usr/bin/env python3
"""Build the Divvy processed-data package submitted with the article."""

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

from bwar.paper_jcgs.divvy_data import (  # noqa: E402
    _initial_fit_raw_end,
    _select_variable_indices,
    _standardized_stream_with_profile,
)
from bwar.paper_jcgs.rolling_origin import make_rolling_origin_splits  # noqa: E402


MONTHS = tuple(f"2024{month:02d}" for month in range(1, 13))
WINDOW = 72
STEP = 24
DIMENSION = 30
MAX_MATRICES = 2000
MAX_ORIGINS = 6
SOURCE_MANIFEST = ROOT / "data" / "divvy_source_manifest.json"
REFERENCE_RESULTS = ROOT / "results" / "reference" / "divvy"
DEFAULT_CACHE = ROOT / "data" / "divvy" / f"hourly_{'_'.join(MONTHS)}.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_strings(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.strftime("%Y-%m-%dT%H:%M:%S"), dtype="string")


def _validate_source_frame(frame: pd.DataFrame) -> None:
    expected_index = pd.date_range(
        "2024-01-01 00:00:00", "2024-12-31 23:00:00", freq="h"
    )
    if not frame.index.equals(expected_index):
        raise ValueError("hourly cache does not cover every hour of calendar year 2024")
    if frame.columns.has_duplicates:
        raise ValueError("hourly cache contains duplicate station identifiers")
    values = frame.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("hourly cache contains non-finite or negative counts")
    if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-12):
        raise ValueError("hourly cache contains non-integer trip counts")


def _source_archive_audit(source_dir: Path | None) -> list[dict[str, object]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for entry in manifest["sources"]:
        row = dict(entry)
        row["verified_locally"] = False
        if source_dir is not None:
            archive = source_dir / f"{entry['month']}.zip"
            if not archive.exists():
                raise FileNotFoundError(f"missing official source archive: {archive}")
            actual = sha256(archive)
            if actual != entry["sha256"]:
                raise ValueError(f"checksum mismatch for {archive.name}")
            if archive.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"byte-size mismatch for {archive.name}")
            row["verified_locally"] = True
        rows.append(row)
    return rows


def _station_selection_audit(
    frame: pd.DataFrame,
    *,
    fit_raw_end: int,
    selected_indices: np.ndarray,
) -> pd.DataFrame:
    values = frame.clip(lower=0.0).to_numpy(float)
    early = values[:fit_raw_end]
    nonzero_fraction = np.nanmean(early > 0.0, axis=0)
    variability = np.nanstd(early, axis=0)
    scores = np.nan_to_num(
        nonzero_fraction * variability,
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.inf,
    )
    order = np.argsort(scores)[::-1]
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    selected = np.zeros(len(frame.columns), dtype=bool)
    selected[selected_indices] = True
    return pd.DataFrame(
        {
            "station_id": [str(value) for value in frame.columns],
            "selection_rank": rank,
            "selected_for_analysis": selected,
            "fit_nonzero_fraction": nonzero_fraction,
            "fit_standard_deviation": variability,
            "activity_variability_score": scores,
        }
    ).sort_values("selection_rank")


def _write_hourly_counts(
    path: Path,
    *,
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
) -> pd.DataFrame:
    selected = frame.iloc[:, selected_indices].copy()
    selected.columns = [str(value) for value in selected.columns]
    selected = selected.reset_index(names="timestamp")
    selected["timestamp"] = _timestamp_strings(pd.DatetimeIndex(selected["timestamp"]))
    for column in selected.columns[1:]:
        selected[column] = np.rint(selected[column].to_numpy(float)).astype(np.int64)
    selected.to_csv(path, index=False)
    return selected


def _write_gaussian_means(
    path: Path,
    *,
    means: np.ndarray,
    starts: np.ndarray,
    time_index: pd.DatetimeIndex,
    station_ids: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index, start in enumerate(starts):
        rows.append(
            pd.DataFrame(
                {
                    "distribution_index": index,
                    "window_start": time_index[int(start)].strftime("%Y-%m-%dT%H:%M:%S"),
                    "window_end_exclusive": (
                        time_index[int(start)] + pd.Timedelta(hours=WINDOW)
                    ).strftime("%Y-%m-%dT%H:%M:%S"),
                    "station_rank": np.arange(1, len(station_ids) + 1),
                    "station_id": station_ids,
                    "standardized_mean": means[index],
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(path, index=False)
    return output


def _write_gaussian_covariances(
    path: Path,
    *,
    covariances: np.ndarray,
    starts: np.ndarray,
    time_index: pd.DatetimeIndex,
    station_ids: tuple[str, ...],
) -> pd.DataFrame:
    row_index, column_index = np.triu_indices(len(station_ids))
    rows: list[pd.DataFrame] = []
    for index, start in enumerate(starts):
        rows.append(
            pd.DataFrame(
                {
                    "distribution_index": index,
                    "window_start": time_index[int(start)].strftime("%Y-%m-%dT%H:%M:%S"),
                    "row_station_rank": row_index + 1,
                    "column_station_rank": column_index + 1,
                    "row_station_id": np.asarray(station_ids, dtype=object)[row_index],
                    "column_station_id": np.asarray(station_ids, dtype=object)[column_index],
                    "standardized_covariance": covariances[index, row_index, column_index],
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(path, index=False)
    return output


def _write_splits(
    path: Path,
    *,
    starts: np.ndarray,
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    splits = make_rolling_origin_splits(len(starts), max_origins=MAX_ORIGINS)
    rows: list[dict[str, object]] = []
    for origin, (fit_end, validation_end, test_end) in enumerate(splits):
        rows.append(
            {
                "origin": origin,
                "fit_end_exclusive": fit_end,
                "validation_end_exclusive": validation_end,
                "test_end_exclusive": test_end,
                "fit_last_window_start": time_index[int(starts[fit_end - 1])].strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "validation_first_window_start": time_index[int(starts[fit_end])].strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "test_first_window_start": time_index[int(starts[validation_end])].strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "test_last_window_start": time_index[int(starts[test_end - 1])].strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False)
    return output


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Divvy processed data for the BWAR study

## Scope

This package contains the processed Divvy inputs used in the article's
rolling-origin application. It does not redistribute the original trip-level
records. The original records remain available from the Divvy System Data
page and are governed by the Divvy Data License Agreement.

## Files

- `divvy_2024_selected_hourly_counts.csv`: hourly departure counts for the 30
  stations used in the analysis. `timestamp` is local Chicago civil time as
  represented in the source archives; all other columns are station IDs and
  contain nonnegative integer trip counts.
- `divvy_2024_gaussian_means.csv`: training-standardized 72-hour window means
  in tidy format. Windows start every 24 hours.
- `divvy_2024_gaussian_covariances.csv`: upper triangles of the corresponding
  30-by-30 covariance matrices after adding the prespecified ridge and applying
  the numerical SPD projection. Symmetry reconstructs the lower triangle.
- `selected_station_profile.csv`: selected station order plus training-period
  centers, scales, activity fractions, variability, and selection scores.
- `all_station_selection_scores.csv`: the leakage-free station-selection audit
  for all candidate station IDs, computed only from the first fit block.
- `rolling_origin_splits.csv`: the five chronological fit/validation/test
  partitions used in the reported Divvy analysis; all end indices are exclusive.
- `divvy_source_manifest.json`: official monthly source URLs, byte sizes, and
  SHA-256 digests.
- `processed_data_manifest.json`: processing parameters, provenance, file row
  counts, and SHA-256 digests.
- `verification_report.json`: checks against the article's archived reference
  protocol and station profile.
- `DATA_LICENSE.md`: source-data rights and attribution information.
- `checksums.sha256`: integrity checks for the package contents.

## Processing and units

The official January--December 2024 trip files were aggregated by trip start
hour and start-station ID. Missing station-hour combinations were filled with
zero. The 30 stations were ranked by the product of their nonzero-hour fraction
and hourly-count standard deviation, using only the first raw fit block (3,096
hours). Centers and scales were also fixed from those same 3,096 hours. The
standardized hourly counts were summarized in 72-hour windows beginning every
24 hours, yielding 364 Gaussian observations. Covariances use NumPy's sample
covariance convention (`ddof=1`) plus a `1e-5` diagonal ridge before SPD
projection.

## Reproduction

The package is generated by `scripts/prepare_divvy_supporting_data.py` in the
BWAR reproducibility repository. Starting from the official monthly archives,
`scripts/download_divvy.py` rebuilds the hourly cache; the preparation script
then creates and verifies this package. The paper's model fitting and inference
are implemented in `scripts/run_divvy.py`. After extracting the package, run

```bash
PYTHONPATH=src python scripts/run_divvy.py \\
  --processed-data-dir /path/to/Divvy_processed_data \\
  --output-dir results/generated/divvy
```

to reproduce the reported rolling-origin analysis directly from the processed
hourly counts.
""",
        encoding="utf-8",
    )


def _write_data_license(path: Path) -> None:
    path.write_text(
        """# Data source, attribution, and licence

The files in this package are derived from the City of Chicago's Divvy system
data made available by Lyft Bikes and Scooters, LLC.

- Source: https://divvybikes.com/system-data
- Data License Agreement: https://divvybikes.com/data-license-agreement
- Licence contact: bike-data@lyft.com

The Divvy agreement permits the data to be included as source material in
analyses, reports, or studies distributed for non-commercial purposes, while
prohibiting hosting or distribution of the data as a stand-alone dataset. This
processed package is supplied only as Supporting Information for the associated
non-commercial scholarly study. It does not include the original trip-level
archives.

The processed data are not covered by the MIT software licence of the code
repository. Users remain responsible for complying with the current Divvy Data
License Agreement and for attributing the original source. No affiliation with
or endorsement by Divvy, Lyft Bikes and Scooters, LLC, or the City of Chicago
is implied.
""",
        encoding="utf-8",
    )


def _verify_against_reference(
    *,
    station_profile: pd.DataFrame,
    splits: pd.DataFrame,
    n_distributions: int,
) -> dict[str, object]:
    reference_stations = pd.read_csv(
        REFERENCE_RESULTS / "selected_stations.csv", dtype={"station_id": "string"}
    )
    station_ids_match = station_profile["station_id"].astype("string").reset_index(
        drop=True
    ).equals(reference_stations["station_id"].reset_index(drop=True))
    centers_match = np.allclose(
        station_profile["training_center"],
        reference_stations["training_center"],
        rtol=0.0,
        atol=1e-14,
    )
    scales_match = np.allclose(
        station_profile["training_scale"],
        reference_stations["training_scale"],
        rtol=0.0,
        atol=1e-14,
    )

    target_panel = pd.read_csv(REFERENCE_RESULTS / "target_level_losses.csv")
    reference_splits = (
        target_panel[["origin", "fit_end", "val_end", "test_end"]]
        .drop_duplicates()
        .sort_values("origin")
        .reset_index(drop=True)
    )
    generated_splits = splits[
        [
            "origin",
            "fit_end_exclusive",
            "validation_end_exclusive",
            "test_end_exclusive",
        ]
    ].rename(
        columns={
            "fit_end_exclusive": "fit_end",
            "validation_end_exclusive": "val_end",
            "test_end_exclusive": "test_end",
        }
    )
    splits_match = generated_splits.equals(reference_splits)

    protocol = json.loads((REFERENCE_RESULTS / "protocol.json").read_text(encoding="utf-8"))
    protocol_match = (
        protocol["stations"] == DIMENSION
        and protocol["window_hours"] == WINDOW
        and protocol["step_hours"] == STEP
        and protocol["rolling_origins"] == len(splits)
    )
    checks = {
        "selected_station_ids_match_article_reference": bool(station_ids_match),
        "training_centers_match_article_reference": bool(centers_match),
        "training_scales_match_article_reference": bool(scales_match),
        "rolling_splits_match_target_level_evidence": bool(splits_match),
        "processing_protocol_matches_article_reference": bool(protocol_match),
        "n_distributional_observations": int(n_distributions),
    }
    checks["passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key != "n_distributional_observations"
    )
    if not checks["passed"]:
        raise ValueError(f"processed data failed article-reference checks: {checks}")
    return checks


def build_package(
    *,
    hourly_cache: Path,
    output_dir: Path,
    source_dir: Path | None,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_audit = _source_archive_audit(source_dir)
    frame = pd.read_parquet(hourly_cache)
    _validate_source_frame(frame)
    fit_raw_end = _initial_fit_raw_end(
        n_raw=len(frame), window=WINDOW, step=STEP, max_matrices=MAX_MATRICES
    )
    values = frame.clip(lower=0.0).to_numpy(float)
    selected_indices = _select_variable_indices(
        values, DIMENSION, early_end=fit_raw_end
    )
    station_ids = tuple(str(frame.columns[index]) for index in selected_indices)
    selected_matrix = values[:, selected_indices]

    means, covariances, _, starts, profile = _standardized_stream_with_profile(
        selected_matrix,
        window=WINDOW,
        step=STEP,
        max_matrices=MAX_MATRICES,
        metric="divvy_standardized_raw_mean_rmse",
        label="Divvy training-standardized station-demand mean RMSE",
        fit_raw_end=fit_raw_end,
    )

    selection_audit = _station_selection_audit(
        frame, fit_raw_end=fit_raw_end, selected_indices=selected_indices
    )
    selection_audit.to_csv(output_dir / "all_station_selection_scores.csv", index=False)
    selected_audit = (
        selection_audit.loc[selection_audit["selected_for_analysis"]]
        .sort_values("selection_rank")
        .reset_index(drop=True)
    )
    station_profile = selected_audit.copy()
    station_profile.insert(0, "station_rank", np.arange(1, DIMENSION + 1))
    station_profile["training_center"] = np.asarray(profile["center"], dtype=float)
    station_profile["training_scale"] = np.asarray(profile["scale"], dtype=float)
    station_profile.to_csv(output_dir / "selected_station_profile.csv", index=False)

    hourly = _write_hourly_counts(
        output_dir / "divvy_2024_selected_hourly_counts.csv",
        frame=frame,
        selected_indices=selected_indices,
    )
    gaussian_means = _write_gaussian_means(
        output_dir / "divvy_2024_gaussian_means.csv",
        means=means,
        starts=starts,
        time_index=pd.DatetimeIndex(frame.index),
        station_ids=station_ids,
    )
    gaussian_covariances = _write_gaussian_covariances(
        output_dir / "divvy_2024_gaussian_covariances.csv",
        covariances=covariances,
        starts=starts,
        time_index=pd.DatetimeIndex(frame.index),
        station_ids=station_ids,
    )
    splits = _write_splits(
        output_dir / "rolling_origin_splits.csv",
        starts=starts,
        time_index=pd.DatetimeIndex(frame.index),
    )
    shutil.copy2(SOURCE_MANIFEST, output_dir / "divvy_source_manifest.json")
    _write_readme(output_dir / "README.md")
    _write_data_license(output_dir / "DATA_LICENSE.md")

    verification = _verify_against_reference(
        station_profile=station_profile,
        splits=splits,
        n_distributions=len(starts),
    )
    verification.update(
        {
            "source_archives_verified_locally": all(
                bool(row["verified_locally"]) for row in source_audit
            ),
            "hourly_counts_are_nonnegative_integers": True,
            "covariance_matrices_are_symmetric_positive_definite": bool(
                np.all(np.linalg.eigvalsh(covariances) > 0.0)
            ),
        }
    )
    verification["passed"] = bool(
        verification["passed"]
        and verification["hourly_counts_are_nonnegative_integers"]
        and verification["covariance_matrices_are_symmetric_positive_definite"]
    )
    (output_dir / "verification_report.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )

    generated_files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"processed_data_manifest.json", "checksums.sha256"}
    ]
    manifest = {
        "dataset_title": "Processed Divvy data for Bures--Wasserstein autoregression",
        "resource_type": "Supporting data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "Divvy public trip records",
            "source_page": "https://divvybikes.com/system-data",
            "license": "https://divvybikes.com/data-license-agreement",
            "period": "2024-01-01 through 2024-12-31",
            "hourly_cache_sha256": sha256(hourly_cache),
            "archives": source_audit,
        },
        "processing": {
            "station_count": DIMENSION,
            "station_selection_block_hours": int(fit_raw_end),
            "station_selection_score": "nonzero-hour fraction times hourly-count standard deviation",
            "standardization_block_hours": int(profile["fit_raw_end"]),
            "window_hours": WINDOW,
            "step_hours": STEP,
            "covariance_ddof": 1,
            "covariance_diagonal_ridge": 1e-5,
            "n_distributional_observations": int(len(starts)),
            "rolling_origins": int(len(splits)),
            "preparation_script": {
                "path": "scripts/prepare_divvy_supporting_data.py",
                "sha256": sha256(Path(__file__)),
            },
        },
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": {
                    "divvy_2024_selected_hourly_counts.csv": len(hourly),
                    "divvy_2024_gaussian_means.csv": len(gaussian_means),
                    "divvy_2024_gaussian_covariances.csv": len(gaussian_covariances),
                    "selected_station_profile.csv": len(station_profile),
                    "all_station_selection_scores.csv": len(selection_audit),
                    "rolling_origin_splits.csv": len(splits),
                }.get(path.name),
            }
            for path in generated_files
        ],
    }
    (output_dir / "processed_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    checksum_files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and verify the Divvy processed Supporting Data package."
    )
    parser.add_argument("--hourly-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = build_package(
        hourly_cache=args.hourly_cache,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
    )
    print(f"Processed Divvy package: {args.output_dir}")
    print(f"Submission archive: {archive}")
    print(f"Archive SHA-256: {sha256(archive)}")


if __name__ == "__main__":
    main()
