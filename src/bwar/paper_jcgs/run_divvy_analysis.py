from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.divvy_data import (  # noqa: E402
    _standardized_stream_with_profile,
    divvy_matrix,
)
from bwar.paper_jcgs.divvy_target_level import (  # noqa: E402
    evaluate_target_panel,
    method_level_summary,
    paired_inference,
)
import bwar.paper_jcgs.divvy_artifacts as rr  # noqa: E402
import bwar.paper_jcgs.rolling_origin as rob  # noqa: E402


DATASET = "divvy"
DISPLAY_DATASET = "Divvy"
MONTHS = tuple(f"2024{month:02d}" for month in range(1, 13))
WINDOW = 72
STEP = 24
DIMENSION = 30
HORIZONS = (3, 4, 5)
MAX_MATRICES = 2000
MAX_ORIGINS = 6
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SENSITIVITY_BLOCK_LENGTH = 6
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_714
DEFAULT_OUT = ROOT / "results" / "generated" / "divvy"
TARGET_METHODS = (
    "persistence",
    "raw_var_window_ar",
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "fixed_bwar",
    "local_bwar",
)


def write_target_level_outputs(
    *,
    output_dir: Path,
    panel: pd.DataFrame,
    tuning: pd.DataFrame,
    selected: pd.DataFrame,
    inference: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "target_level_losses.csv", index=False)
    inference.to_csv(output_dir / "paired_inference.csv", index=False)
    tuning.to_csv(output_dir / "local_tuning.csv", index=False)
    selected.to_csv(output_dir / "local_selected_settings.csv", index=False)
    diagnostic_columns = [
        column
        for column in (
            "origin",
            "horizon",
            "source_index",
            "target_index",
            "method",
            "window_length",
            "ridge",
            "min_pred_eig",
            "reference_residual",
            "reference_refreshed",
            "reference_fallback",
        )
        if column in panel.columns
    ]
    diagnostics = panel.loc[panel["method"].eq("local_bwar"), diagnostic_columns]
    diagnostics.to_csv(output_dir / "local_reference_diagnostics.csv", index=False)


def build_target_level_evidence(
    *,
    means: np.ndarray,
    covariances: np.ndarray,
    raw_series: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    splits: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    horizons: tuple[int, ...],
    domain_profile: dict[str, object],
    ridge_grid: tuple[float, ...],
    local_window_grid: tuple[int, ...] = (24, 48, 72),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, tuning, selected = evaluate_target_panel(
        means,
        covariances,
        raw_series=raw_series,
        window_starts=starts,
        window_size=window_size,
        splits=splits,
        horizons=horizons,
        domain_profile=domain_profile,
        ridge_grid=ridge_grid,
        local_window_grid=local_window_grid,
    )
    inference = paired_inference(
        panel,
        targets=("fixed_bwar", "local_bwar"),
        comparators=(
            "persistence",
            "raw_var_window_ar",
            "euclidean_gaussian_ar",
            "cholesky_gaussian_ar",
            "log_euclidean_gaussian_ar",
        ),
        horizon=int(horizons[0]),
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        sensitivity_block_length=BOOTSTRAP_SENSITIVITY_BLOCK_LENGTH,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    local_vs_fixed = paired_inference(
        panel,
        targets=("local_bwar",),
        comparators=("fixed_bwar",),
        horizon=int(horizons[0]),
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        sensitivity_block_length=BOOTSTRAP_SENSITIVITY_BLOCK_LENGTH,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED + 50_000,
    )
    inference = pd.concat([inference, local_vs_fixed], ignore_index=True)
    return panel, tuning, selected, inference


def add_run_metadata(
    frame: pd.DataFrame,
    *,
    dimension: int,
    n_matrices: int,
    primary_horizon: int,
) -> pd.DataFrame:
    decorated = frame.copy()
    decorated["dimension"] = int(dimension)
    decorated["dimension_arg"] = int(dimension)
    decorated["n_matrices"] = int(n_matrices)
    decorated["h0"] = int(primary_horizon)
    decorated["q_dimension"] = int(
        dimension + dimension * (dimension + 1) // 2
    )
    return decorated


def first_held_out_forecast_indices(
    *, validation_end: int, horizon: int
) -> tuple[int, int]:
    """Return the prespecified first forecast origin and target in test block 1."""
    if horizon < 1 or validation_end < horizon:
        raise ValueError("validation_end must be at least the positive horizon")
    return validation_end - horizon, validation_end


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def write_representative_forecast_snapshot(
    *,
    output_dir: Path,
    station_ids: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
    forecasts: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, object],
) -> None:
    """Write tidy, traceable domain evidence for one prespecified forecast."""
    output_dir.mkdir(parents=True, exist_ok=True)
    center = np.asarray(center, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if len(station_ids) != len(center) or len(scale) != len(center):
        raise ValueError("station ids, center, and scale must have matching lengths")

    mean_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    scaling = np.diag(scale)
    for method, (mean, covariance) in forecasts.items():
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if mean.shape != center.shape or covariance.shape != (len(center), len(center)):
            raise ValueError(f"forecast dimensions do not match for {method}")
        physical_mean = mean * scale + center
        physical_covariance = scaling @ covariance @ scaling
        correlation = _correlation_from_covariance(physical_covariance)
        for station_rank, (station_id, value) in enumerate(
            zip(station_ids, physical_mean, strict=True), start=1
        ):
            mean_rows.append(
                {
                    "method": method,
                    "station_rank": station_rank,
                    "station_id": station_id,
                    "mean_count": float(value),
                }
            )
        for row in range(len(center)):
            for column in range(len(center)):
                correlation_rows.append(
                    {
                        "method": method,
                        "row": row,
                        "column": column,
                        "correlation": float(correlation[row, column]),
                    }
                )

    pd.DataFrame(mean_rows).to_csv(
        output_dir / "representative_forecast_means.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "representative_forecast_correlations.csv", index=False
    )
    (output_dir / "representative_forecast_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def fit_representative_forecasts(
    *,
    means: np.ndarray,
    covariances: np.ndarray,
    fit_end: int,
    validation_end: int,
    horizon: int,
    domain_profile: dict[str, object],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Fit the two displayed methods and forecast the prespecified first target."""
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    dimension = means.shape[1]
    forecast_origin, target_index = first_held_out_forecast_indices(
        validation_end=validation_end, horizon=horizon
    )

    cholesky = rob.select_encoded_model(
        means,
        covariances,
        rob.cholesky_encode,
        lambda z: rob.cholesky_decode(z, dimension),
        fit_end=fit_end,
        val_end=validation_end,
        horizon=horizon,
        ar_model="full",
        selection_metric="domain",
        domain_profile=domain_profile,
    )
    cholesky_z = rob.recursive_predict_z(
        cholesky["Z"][forecast_origin], cholesky["W"], horizon
    )
    cholesky_forecast = cholesky["decode"](cholesky_z)

    reference_name, reference_mean, reference_covariance = (
        rr.barycenter_only_references(means[:fit_end], covariances[:fit_end])[0]
    )
    bwar = rob.select_encoded_model(
        means,
        covariances,
        lambda mean, covariance: rob.bwar_gaussian_encode(
            mean, covariance, reference_mean, reference_covariance
        ),
        lambda z: rob.bwar_gaussian_decode(z, reference_mean, reference_covariance),
        fit_end=fit_end,
        val_end=validation_end,
        horizon=horizon,
        ar_model="full",
        selection_metric="domain",
        domain_profile=domain_profile,
    )
    bwar_z = rob.recursive_predict_z(
        bwar["Z"][forecast_origin], bwar["W"], horizon
    )
    bwar_forecast = bwar["decode"](bwar_z)

    forecasts = {
        "observed": (means[target_index], covariances[target_index]),
        "cholesky_gaussian_ar": cholesky_forecast,
        "bwar_barycenter": bwar_forecast,
    }
    metadata = {
        "selection_rule": "first target in the first held-out block",
        "forecast_origin": int(forecast_origin),
        "target_index": int(target_index),
        "horizon": int(horizon),
        "fit_end": int(fit_end),
        "validation_end": int(validation_end),
        "reference": reference_name,
        "cholesky_ridge": float(cholesky["ridge"]),
        "bwar_ridge": float(bwar["ridge"]),
    }
    return forecasts, metadata


def protocol_job_name(*, window: int, step: int, dimension: int) -> str:
    return f"divvy_2024_w{window}_s{step}_d{dimension}"


def monthly_h2_splits(
    starts: np.ndarray,
    *,
    window: int,
    series_start: str = "2024-01-01 00:00:00",
) -> list[tuple[int, int, int]]:
    start_times = pd.DatetimeIndex(pd.Timestamp(series_start) + pd.to_timedelta(starts, unit="h"))
    end_times = start_times + pd.to_timedelta(window, unit="h")
    splits: list[tuple[int, int, int]] = []
    for month in range(7, 13):
        test_start = pd.Timestamp(year=2024, month=month, day=1)
        test_stop = test_start + pd.DateOffset(months=1)
        fit_cutoff = test_start - pd.DateOffset(months=2)
        fit_end = int(end_times.searchsorted(fit_cutoff, side="right"))
        val_end = int(start_times.searchsorted(test_start, side="left"))
        test_end = min(len(starts), int(start_times.searchsorted(test_stop, side="left")))
        splits.append((fit_end, val_end, test_end))
    return splits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_application_case(summary: pd.DataFrame) -> pd.Series:
    key = (
        summary["candidate"].eq(DATASET)
        & summary["window"].eq(WINDOW)
        & summary["step"].eq(STEP)
        & summary["dimension"].eq(DIMENSION)
        & summary["horizon"].eq(HORIZONS[0])
    )
    selected = summary.loc[key]
    if len(selected) != 1:
        raise ValueError("confirmation summary does not contain exactly one row for the fixed Divvy protocol")
    return selected.iloc[0]


def write_manifest(
    path: Path,
    *,
    station_ids: tuple[str, ...],
    starts: np.ndarray,
    profile: dict[str, object],
    long: pd.DataFrame,
    target_panel: pd.DataFrame | None = None,
    selected_local: pd.DataFrame | None = None,
) -> None:
    split_rows = (
        long[["origin", "fit_end", "val_end", "test_end"]]
        .drop_duplicates()
        .sort_values("origin")
        .to_dict(orient="records")
    )
    cache = ROOT / "data" / DATASET / f"hourly_{'_'.join(MONTHS)}.parquet"
    source_files = [ROOT / "data" / DATASET / f"{month}.zip" for month in MONTHS]
    source_files.append(cache)
    sources = [
        {
            "path": str(source.relative_to(ROOT)),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
        for source in source_files
        if source.exists()
    ]
    target_panel = pd.DataFrame() if target_panel is None else target_panel
    selected_local = pd.DataFrame() if selected_local is None else selected_local
    primary_targets = (
        target_panel.loc[target_panel["horizon"].eq(HORIZONS[0]), "target_index"]
        if {"horizon", "target_index"}.issubset(target_panel.columns)
        else pd.Series(dtype=int)
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "determinism": "deterministic",
        "dataset": DATASET,
        "display_dataset": DISPLAY_DATASET,
        "months": list(MONTHS),
        "window": WINDOW,
        "step": STEP,
        "dimension": DIMENSION,
        "coordinate_dimension": DIMENSION + DIMENSION * (DIMENSION + 1) // 2,
        "horizons": list(HORIZONS),
        "primary_horizon": HORIZONS[0],
        "primary_horizon_rule": "ceil(window / step), the first non-overlapping target window",
        "reference": "fixed Bures barycenter computed from each origin's fit block",
        "ar_model": "full AR(1)",
        "reported_endpoints": [
            "same-task Gaussian W2-squared loss",
            "training-standardized physical-mean RMSE",
        ],
        "evaluation_protocol": str(long["evaluation_protocol"].iloc[0]),
        "station_selection": "top activity-variability score using only the first origin's raw fit block",
        "standardization": "center and scale fixed from the first origin's raw fit block",
        "standardization_center": np.asarray(profile["center"], dtype=float).tolist(),
        "standardization_scale": np.asarray(profile["scale"], dtype=float).tolist(),
        "selected_station_ids": list(station_ids),
        "n_distributional_observations": int(len(starts)),
        "rolling_splits": split_rows,
        "target_level_evaluation": {
            "primary_target_count": int(primary_targets.nunique()),
            "complete_paired_panel": True,
            "loss_file": "target_level_losses.csv",
            "inference_file": "paired_inference.csv",
        },
        "bootstrap": {
            "kind": "origin-preserving moving-block bootstrap",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "sensitivity_block_length": BOOTSTRAP_SENSITIVITY_BLOCK_LENGTH,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "multiplicity": "Holm adjustment within BWAR variant and endpoint",
        },
        "local_reference": {
            "window_grid": [24, 48, 72],
            "refresh_period": 1,
            "selection_endpoint": "validation raw RMSE",
            "selected_settings": selected_local.to_dict(orient="records"),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "sources": sources,
        "code_hashes": {
            str(source.relative_to(ROOT)): sha256(source)
            for source in [
                Path(__file__),
                Path(__file__).with_name("divvy_artifacts.py"),
                Path(__file__).with_name("rolling_origin.py"),
                Path(__file__).with_name("divvy_data.py"),
                Path(__file__).with_name("divvy_target_level.py"),
                Path(__file__).with_name("local_reference_bwar.py"),
            ]
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(out_dir: Path, *, h2_monthly: bool = False) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "artifacts" / "tables"
    figure_dir = out_dir / "artifacts" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fit_raw_end = None
    if h2_monthly:
        if MONTHS != tuple(f"2024{month:02d}" for month in range(1, 13)):
            raise ValueError("monthly H2 confirmation requires January--December 2024")
        fit_raw_end = int((pd.Timestamp("2024-05-01") - pd.Timestamp("2024-01-01")) / pd.Timedelta(hours=1))

    matrix, station_ids = divvy_matrix(
        DIMENSION,
        months=MONTHS,
        window=WINDOW,
        step=STEP,
        max_matrices=MAX_MATRICES,
        fit_raw_end=fit_raw_end,
        return_columns=True,
    )
    means, covs, raw_windows, starts, profile = _standardized_stream_with_profile(
        matrix,
        window=WINDOW,
        step=STEP,
        max_matrices=MAX_MATRICES,
        metric="divvy_standardized_raw_mean_rmse",
        label="Divvy training-standardized station-demand mean RMSE",
        fit_raw_end=fit_raw_end,
    )
    splits = (
        monthly_h2_splits(starts, window=WINDOW)
        if h2_monthly
        else rob.make_rolling_origin_splits(len(covs), max_origins=MAX_ORIGINS)
    )
    meta = {
        "window": WINDOW,
        "step": STEP,
        "months": ",".join(MONTHS),
        "physical_units": "training-standardized",
        "station_selection_block": "first_origin_fit",
        "standardization_block": "first_origin_fit",
        "evaluation_protocol": "monthly_h2_confirmation" if h2_monthly else "fractional_rolling_origin",
    }

    original_reference_builder = rob.candidate_gaussian_references
    rob.candidate_gaussian_references = rr.barycenter_only_references
    try:
        raw, refs = rob.run_rolling_origin_series(
            job=protocol_job_name(window=WINDOW, step=STEP, dimension=DIMENSION)
            + ("_h2monthly" if h2_monthly else ""),
            dataset=DATASET,
            means=means,
            covs=covs,
            meta=meta,
            horizons=list(HORIZONS),
            raw_windows=raw_windows,
            window_starts=starts,
            window_size=WINDOW,
            ar_model="full",
            domain_profile_override=profile,
            max_origins=MAX_ORIGINS,
            splits_override=splits,
        )
    finally:
        rob.candidate_gaussian_references = original_reference_builder

    long = add_run_metadata(
        rr.combine_bwar_rows(raw, refs, candidate=DATASET),
        dimension=DIMENSION,
        n_matrices=len(covs),
        primary_horizon=HORIZONS[0],
    )
    summary, method_summary = rr.summarize_long(long)
    case = fixed_application_case(summary)
    raw_series = rob.reconstruct_raw_series_from_windows(raw_windows, starts)
    target_panel, local_tuning, local_selected, inference = build_target_level_evidence(
        means=means,
        covariances=covs,
        raw_series=raw_series,
        starts=starts,
        window_size=WINDOW,
        splits=tuple(splits),
        horizons=HORIZONS,
        domain_profile=profile,
        ridge_grid=rob.DEFAULT_RIDGE_GRID,
    )
    target_method_summary = method_level_summary(
        target_panel,
        methods=TARGET_METHODS,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    long.to_csv(out_dir / "origin_level_results.csv", index=False)
    summary.to_csv(out_dir / "configuration_summary.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    pd.DataFrame([case.to_dict()]).to_csv(out_dir / "fixed_application.csv", index=False)
    pd.DataFrame(
        {
            "station_id": station_ids,
            "training_center": np.asarray(profile["center"], dtype=float),
            "training_scale": np.asarray(profile["scale"], dtype=float),
        }
    ).to_csv(out_dir / "selected_stations.csv", index=False)
    write_target_level_outputs(
        output_dir=out_dir,
        panel=target_panel,
        tuning=local_tuning,
        selected=local_selected,
        inference=inference,
    )
    target_method_summary.to_csv(
        out_dir / "method_level_bootstrap_summary.csv",
        index=False,
    )

    rr.write_application_table(method_summary, case, table_dir / "redone_realdata_application.tex")
    rr.write_horizon_table(method_summary, case, table_dir / "redone_realdata_horizon.tex")
    rr.make_application_figure(long, method_summary, case, figure_dir / "redone_realdata_application")
    write_manifest(
        out_dir / "protocol_manifest.json",
        station_ids=station_ids,
        starts=starts,
        profile=profile,
        long=long,
        target_panel=target_panel,
        selected_local=local_selected,
    )

    key = (
        method_summary["candidate"].eq(DATASET)
        & method_summary["window"].eq(WINDOW)
        & method_summary["step"].eq(STEP)
        & method_summary["dimension"].eq(DIMENSION)
        & method_summary["horizon"].eq(HORIZONS[0])
    )
    print("Prespecified Divvy real-data protocol completed.")
    print(f"Output: {out_dir}")
    print(f"Distributions: {len(covs)}; rolling origins: {long['origin'].nunique()}")
    print(
        f"Primary target count: "
        f"{target_panel.loc[target_panel['horizon'].eq(HORIZONS[0]), 'target_index'].nunique()}"
    )
    print("Selected local settings:")
    print(local_selected.to_string(index=False))
    print(method_summary.loc[key, ["method", "raw_mean", "raw_se", "w2_mean", "w2_se", "n_origins"]].sort_values("raw_mean").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prespecified Divvy real-data application."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--h2-monthly", action="store_true")
    args = parser.parse_args()
    run(args.out_dir, h2_monthly=args.h2_monthly)


if __name__ == "__main__":
    main()
