#!/usr/bin/env python3
"""Frozen-protocol PEMS-BAY Gaussian-state comparison for the BWAR paper.

The script deliberately separates the *method-blind* construction of spatial
panels, covariance regularisation, and temporal splits from the final test
comparison.  PEMS is represented as non-overlapping 12-hour blocks of five-
minute sensor speeds, hence every Gaussian state uses 144 observations for
20 dimensions.  No raw observations are shared by two consecutive states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from bwar.gaussian_geometry import bw2_cov, bw_barycenter, project_spd
from bwar.paper_jcgs.gaussian_models import (
    bwar_gaussian_decode,
    bwar_gaussian_encode,
    cholesky_decode,
    cholesky_encode,
    euclidean_decode,
    euclidean_encode,
    fit_var,
    gaussian_w2_squared,
    log_euclidean_decode,
    log_euclidean_encode,
    recursive_predict_z,
)


SEED = 20260731
BLOCK_SIZE = 144  # 12 hours at 5-minute resolution
DIMENSION = 20
N_PANELS = 4
FIT_FRACTION = 0.60
VALIDATION_FRACTION = 0.80
HORIZONS = (1, 3, 6)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SHRINKAGE_GRID = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30)
CONDITION_Q90_CAP = 100.0


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["speed/block0_values"], dtype=float)
        identifiers = np.asarray(handle["speed/axis0"], dtype=int)
        timestamps = np.asarray(handle["speed/axis1"], dtype=np.int64)
    return values, identifiers, timestamps


def _load_locations(path: Path, identifiers: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, header=None, names=["sensor", "latitude", "longitude"])
    by_id = frame.set_index("sensor")[["latitude", "longitude"]]
    locations = by_id.loc[identifiers].to_numpy(float)
    if not np.isfinite(locations).all():
        raise ValueError("all PEMS sensors must have finite locations")
    return locations


def _select_spatial_panels(
    raw: np.ndarray,
    locations: np.ndarray,
    identifiers: np.ndarray,
) -> tuple[list[np.ndarray], list[list[int]], dict[str, float]]:
    """Choose four deterministic spatial panels using training data only."""
    fit_rows = int(np.floor(FIT_FRACTION * len(raw)))
    fitting = raw[:fit_rows]
    valid = np.isfinite(fitting) & (fitting > 0.0)
    coverage = valid.mean(axis=0)
    masked = np.where(valid, fitting, np.nan)
    variability = np.nanstd(masked, axis=0)
    eligible = np.flatnonzero((coverage >= 0.995) & (variability > 1.0))
    if len(eligible) < N_PANELS * DIMENSION:
        raise RuntimeError("fewer than 80 fully usable PEMS sensors in fitting period")

    centers, labels = kmeans2(
        locations[eligible], N_PANELS, minit="++", seed=SEED
    )
    panels: list[np.ndarray] = []
    panel_ids: list[list[int]] = []
    for label in range(N_PANELS):
        members = eligible[labels == label]
        if len(members) < DIMENSION:
            raise RuntimeError("a geographic PEMS cluster has fewer than 20 sensors")
        squared_distance = ((locations[members] - centers[label]) ** 2).sum(axis=1)
        panel = members[np.argsort(squared_distance)[:DIMENSION]]
        panels.append(panel)
        panel_ids.append([int(value) for value in identifiers[panel]])
    audit = {
        "eligible_sensor_count": int(len(eligible)),
        "fitting_min_sensor_coverage": float(coverage[eligible].min()),
        "fitting_median_sensor_sd": float(np.median(variability[eligible])),
    }
    return panels, panel_ids, audit


def _contiguous_block_starts(timestamps: np.ndarray) -> np.ndarray:
    step = np.int64(5 * 60 * 1_000_000_000)
    starts = []
    for start in range(0, len(timestamps) - BLOCK_SIZE + 1, BLOCK_SIZE):
        window = timestamps[start : start + BLOCK_SIZE]
        if np.all(np.diff(window) == step):
            starts.append(start)
    return np.asarray(starts, dtype=int)


def _standardize_from_fitting_period(raw: np.ndarray) -> np.ndarray:
    fit_rows = int(np.floor(FIT_FRACTION * len(raw)))
    fitting = raw[:fit_rows]
    valid = np.isfinite(fitting) & (fitting > 0.0)
    median = np.nanmedian(np.where(valid, fitting, np.nan), axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("training-only PEMS medians are not finite")
    clean = np.where(np.isfinite(raw) & (raw > 0.0), raw, median[None, :])
    center = clean[:fit_rows].mean(axis=0)
    scale = clean[:fit_rows].std(axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise RuntimeError("a selected PEMS sensor has zero training variation")
    return (clean - center) / scale


def _state_moments(values: np.ndarray, starts: np.ndarray, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    means, covariances = [], []
    for start in starts:
        block = values[start : start + BLOCK_SIZE]
        mean = block.mean(axis=0)
        covariance = np.cov(block, rowvar=False, ddof=1)
        diagonal_target = np.eye(covariance.shape[0]) * np.trace(covariance) / covariance.shape[0]
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal_target
        means.append(mean)
        covariances.append(project_spd(covariance, eps=1e-7))
    return np.asarray(means), np.asarray(covariances)


def _condition_q90(covariances: np.ndarray, end: int) -> float:
    values = [np.linalg.cond(covariance) for covariance in covariances[:end]]
    return float(np.quantile(values, 0.90))


def _choose_shrinkage(standardized: np.ndarray, starts: np.ndarray, panels: list[np.ndarray], val_end: int) -> tuple[float, dict[str, float]]:
    candidates: dict[float, list[float]] = {}
    for value in SHRINKAGE_GRID:
        q90s = []
        for panel in panels:
            _, covariances = _state_moments(standardized[:, panel], starts, value)
            q90s.append(_condition_q90(covariances, val_end))
        candidates[value] = q90s
    chosen = next(
        (value for value in SHRINKAGE_GRID if max(candidates[value]) <= CONDITION_Q90_CAP),
        min(SHRINKAGE_GRID, key=lambda value: max(candidates[value])),
    )
    audit = {f"shrinkage_{value:.2f}_worst_q90_condition": float(max(q90s)) for value, q90s in candidates.items()}
    audit["selected_shrinkage"] = float(chosen)
    return float(chosen), audit


def _normalised_commutator(covariances: np.ndarray, end: int) -> np.ndarray:
    output = []
    for previous, current in zip(covariances[: end - 1], covariances[1:end]):
        numerator = np.linalg.norm(previous @ current - current @ previous, ord="fro")
        denominator = np.linalg.norm(previous, ord="fro") * np.linalg.norm(current, ord="fro")
        output.append(numerator / max(denominator, 1e-12))
    return np.asarray(output)


def _adjacent_decomposition(means: np.ndarray, covariances: np.ndarray, end: int) -> dict[str, float]:
    mean_part, covariance_part = [], []
    for index in range(end - 1):
        mean_part.append(float(np.sum((means[index + 1] - means[index]) ** 2)))
        covariance_part.append(float(bw2_cov(covariances[index + 1], covariances[index])))
    total = np.asarray(mean_part) + np.asarray(covariance_part)
    split = end // 2
    early_reference = bw_barycenter(covariances[:split])
    late_reference = bw_barycenter(covariances[split:end])
    return {
        "median_adjacent_w2": float(np.median(total)),
        "covariance_share_of_adjacent_w2": float(np.sum(covariance_part) / max(np.sum(total), 1e-12)),
        "early_late_reference_shift_over_median_adjacent_w2": float(
            bw2_cov(early_reference, late_reference) / max(np.median(total), 1e-12)
        ),
    }


def _covariance_pc_lag1(covariances: np.ndarray, end: int) -> float:
    d = covariances.shape[1]
    upper = np.triu_indices(d)
    coordinates = np.asarray([covariance[upper] for covariance in covariances[:end]])
    centered = coordinates - coordinates.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right[0]
    return float(np.corrcoef(scores[:-1], scores[1:])[0, 1])


def _raw_gaussian_nll(raw_block: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    covariance = project_spd(covariance, eps=1e-7)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise FloatingPointError("decoded covariance is not positive definite")
    delta = raw_block - mean
    solved = np.linalg.solve(covariance, delta.T).T
    quadratic = np.sum(delta * solved, axis=1)
    d = covariance.shape[0]
    return float(0.5 * (d * np.log(2.0 * np.pi) + logdet + np.mean(quadratic)))


def _encode_methods(means: np.ndarray, covariances: np.ndarray, fit_end: int) -> dict[str, tuple[np.ndarray, object]]:
    reference_mean = means[:fit_end].mean(axis=0)
    reference_covariance = bw_barycenter(covariances[:fit_end])
    return {
        "Euclidean": (
            np.asarray([euclidean_encode(mean, covariance) for mean, covariance in zip(means, covariances)]),
            lambda z: euclidean_decode(z, means.shape[1]),
        ),
        "Cholesky": (
            np.asarray([cholesky_encode(mean, covariance) for mean, covariance in zip(means, covariances)]),
            lambda z: cholesky_decode(z, means.shape[1]),
        ),
        "Log-Euclidean": (
            np.asarray([log_euclidean_encode(mean, covariance) for mean, covariance in zip(means, covariances)]),
            lambda z: log_euclidean_decode(z, means.shape[1]),
        ),
        "BWAR": (
            np.asarray(
                [
                    bwar_gaussian_encode(mean, covariance, reference_mean, reference_covariance)
                    for mean, covariance in zip(means, covariances)
                ]
            ),
            lambda z: bwar_gaussian_decode(z, reference_mean, reference_covariance),
        ),
    }


def _w2_over_interval(
    means: np.ndarray,
    covariances: np.ndarray,
    coordinates: np.ndarray,
    decoder: object,
    matrix: np.ndarray,
    start: int,
    stop: int,
    horizon: int,
) -> float:
    losses = []
    for source in range(start, stop - horizon):
        prediction = recursive_predict_z(coordinates[source], matrix, horizon)
        mean, covariance = decoder(prediction)
        losses.append(gaussian_w2_squared(mean, covariance, means[source + horizon], covariances[source + horizon]))
    return float(np.mean(losses))


def _select_ridge(
    means: np.ndarray,
    covariances: np.ndarray,
    coordinates: np.ndarray,
    decoder: object,
    fit_end: int,
    val_end: int,
    horizon: int,
) -> float:
    scores = {}
    for ridge in RIDGE_GRID:
        matrix = fit_var(coordinates, fit_end, lam=ridge, model="diag")
        scores[ridge] = _w2_over_interval(
            means, covariances, coordinates, decoder, matrix, fit_end, val_end, horizon
        )
    return min(scores, key=scores.get)


def _record(
    *, panel: int, source: int, horizon: int, method: str, prediction: tuple[np.ndarray, np.ndarray], target_mean: np.ndarray, target_covariance: np.ndarray, target_block: np.ndarray,
) -> dict[str, float | int | str]:
    mean, covariance = prediction
    mean_w2 = float(np.sum((mean - target_mean) ** 2))
    covariance_w2 = float(bw2_cov(covariance, target_covariance))
    return {
        "panel": panel,
        "source_state": source,
        "horizon": horizon,
        "method": method,
        "w2_squared": mean_w2 + covariance_w2,
        "mean_w2_component": mean_w2,
        "covariance_w2_component": covariance_w2,
        "mean_rmse": float(np.sqrt(np.mean((mean - target_mean) ** 2))),
        "raw_gaussian_nll": _raw_gaussian_nll(target_block, mean, covariance),
    }


def _evaluate_panel(
    panel_number: int,
    raw_panel: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    fit_end: int,
    val_end: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    encodings = _encode_methods(means, covariances, fit_end)
    selected: list[dict[str, float | int | str]] = []
    matrices: dict[tuple[str, int], tuple[np.ndarray, object]] = {}
    for name, (coordinates, decoder) in encodings.items():
        for horizon in HORIZONS:
            ridge = _select_ridge(means, covariances, coordinates, decoder, fit_end, val_end, horizon)
            matrices[(name, horizon)] = (fit_var(coordinates, val_end, lam=ridge, model="diag"), decoder)
            selected.append({"panel": panel_number, "method": name, "horizon": horizon, "ridge": ridge})

    records: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        for source in range(val_end, len(means) - horizon):
            target = source + horizon
            target_block = raw_panel[target]
            for name, (matrix, decoder) in ((key[0], value) for key, value in matrices.items() if key[1] == horizon):
                prediction = decoder(recursive_predict_z(encodings[name][0][source], matrix, horizon))
                records.append(
                    _record(
                        panel=panel_number, source=source, horizon=horizon, method=name,
                        prediction=prediction, target_mean=means[target], target_covariance=covariances[target], target_block=target_block,
                    )
                )
            records.append(
                _record(
                    panel=panel_number, source=source, horizon=horizon, method="Persistence",
                    prediction=(means[source], covariances[source]), target_mean=means[target], target_covariance=covariances[target], target_block=target_block,
                )
            )
    return records, selected


def _moving_block_summary(
    values: np.ndarray, *, rng: np.random.Generator, block: int = 7, draws: int = 2000
) -> tuple[float, float, float]:
    """Bootstrap standard error and interval for an ordered test-period mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, count))
    offsets = np.arange(block)[None, None, :]
    indices = (starts[:, :, None] + offsets) % n
    samples = values[indices].reshape(draws, -1)[:, :n].mean(axis=1)
    standard_error = float(np.std(samples, ddof=1))
    low, high = np.quantile(samples, [0.025, 0.975])
    return standard_error, float(low), float(high)


def _paired_effects(records: pd.DataFrame) -> pd.DataFrame:
    """BWAR minus comparator effects, paired by time and averaged over panels."""
    rows = []
    rng = np.random.default_rng(SEED)
    comparators = ("Euclidean", "Cholesky", "Log-Euclidean", "Persistence")
    for horizon in HORIZONS:
        period = records[records.horizon == horizon]
        for metric in ("w2_squared", "covariance_w2_component"):
            wide = period.pivot_table(index=["source_state", "panel"], columns="method", values=metric)
            for comparator in comparators:
                contrast = (wide["BWAR"] - wide[comparator]).groupby(level="source_state").mean().sort_index().to_numpy()
                _, low, high = _moving_block_summary(contrast, rng=rng)
                rows.append({
                    "horizon": horizon, "metric": metric, "comparison": f"BWAR minus {comparator}",
                    "mean_difference": float(contrast.mean()), "ci_2_5": low, "ci_97_5": high,
                    "n_time_points": len(contrast),
                })
    return pd.DataFrame(rows)


def _method_intervals(records: pd.DataFrame) -> pd.DataFrame:
    """Method-level moving-block intervals, clustered over geographic panels."""
    rows = []
    rng = np.random.default_rng(SEED + 8107)
    metrics = ("w2_squared", "covariance_w2_component", "mean_rmse")
    for horizon in HORIZONS:
        period = records[records.horizon == horizon]
        for method in sorted(period.method.unique()):
            one_method = period[period.method == method]
            for metric in metrics:
                ordered = (
                    one_method.groupby("source_state", as_index=True)[metric]
                    .mean()
                    .sort_index()
                    .to_numpy()
                )
                standard_error, low, high = _moving_block_summary(ordered, rng=rng)
                rows.append(
                    {
                        "method": method,
                        "horizon": horizon,
                        "metric": metric,
                        "standard_error": standard_error,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/pems_bay/pems-bay.h5"))
    parser.add_argument("--locations", type=Path, default=Path("data/pems_bay/graph_sensor_locations_bay.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/pems_bay_primary"))
    args = parser.parse_args()
    np.random.seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)

    raw, identifiers, timestamps = _load_raw(args.raw)
    locations = _load_locations(args.locations, identifiers)
    panels, panel_ids, sensor_audit = _select_spatial_panels(raw, locations, identifiers)
    starts = _contiguous_block_starts(timestamps)
    n_states = len(starts)
    fit_end = int(np.floor(FIT_FRACTION * n_states))
    val_end = int(np.floor(VALIDATION_FRACTION * n_states))
    if n_states < 250 or n_states - val_end < 50:
        raise RuntimeError(f"PEMS has only {n_states} usable states / {n_states - val_end} test sources")
    standardized = _standardize_from_fitting_period(raw)
    shrinkage, condition_audit = _choose_shrinkage(standardized, starts, panels, val_end)

    protocol = {
        "dataset": "PEMS-BAY", "raw_md5": _md5(args.raw), "seed": SEED,
        "state_construction": "non-overlapping 12-hour blocks of five-minute readings",
        "block_size": BLOCK_SIZE, "dimension": DIMENSION, "panel_count": N_PANELS,
        "n_states": n_states, "fit_end": fit_end, "validation_end": val_end,
        "horizons": list(HORIZONS), "ar_model": "diagonal ridge VAR(1)",
        "ridge_grid": list(RIDGE_GRID),
        "panel_sensor_ids": panel_ids, "selected_covariance_shrinkage": shrinkage,
        "sensor_audit": sensor_audit, "condition_audit": condition_audit,
    }
    (args.out / "protocol_lock.json").write_text(json.dumps(protocol, indent=2) + "\n")

    diagnostics = []
    all_records: list[dict[str, float | int | str]] = []
    all_ridges: list[dict[str, float | int | str]] = []
    for number, panel in enumerate(panels, start=1):
        means, covariances = _state_moments(standardized[:, panel], starts, shrinkage)
        decomposed = _adjacent_decomposition(means, covariances, val_end)
        commutator = _normalised_commutator(covariances, val_end)
        diagnostics.append({
            "panel": number,
            "median_normalised_commutator": float(np.median(commutator)),
            "q75_normalised_commutator": float(np.quantile(commutator, 0.75)),
            "q90_condition": _condition_q90(covariances, val_end),
            "covariance_pc1_lag1_autocorrelation": _covariance_pc_lag1(covariances, val_end),
            **decomposed,
        })
        raw_blocks = np.asarray([standardized[start : start + BLOCK_SIZE, panel] for start in starts])
        records, ridges = _evaluate_panel(number, raw_blocks, means, covariances, fit_end, val_end)
        all_records.extend(records)
        all_ridges.extend(ridges)

    record_frame = pd.DataFrame(all_records)
    record_frame.to_csv(args.out / "test_origin_level.csv", index=False)
    pd.DataFrame(all_ridges).to_csv(args.out / "validation_selected_ridges.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out / "training_validation_geometry_audit.csv", index=False)
    summary = record_frame.groupby(["method", "horizon"], as_index=False).agg(
        w2_squared_mean=("w2_squared", "mean"), w2_squared_median=("w2_squared", "median"),
        covariance_w2_component_mean=("covariance_w2_component", "mean"),
        mean_rmse_mean=("mean_rmse", "mean"), raw_gaussian_nll_mean=("raw_gaussian_nll", "mean"),
        n=("w2_squared", "size"),
    )
    intervals = _method_intervals(record_frame)
    for metric in ("w2_squared", "covariance_w2_component", "mean_rmse"):
        subset = intervals[intervals.metric == metric][["method", "horizon", "standard_error", "ci_low", "ci_high"]].rename(
            columns={
                "standard_error": f"{metric}_standard_error",
                "ci_low": f"{metric}_ci_low",
                "ci_high": f"{metric}_ci_high",
            }
        )
        summary = summary.merge(subset, on=["method", "horizon"], how="left", validate="one_to_one")
    summary.to_csv(args.out / "test_method_summary.csv", index=False)
    _paired_effects(record_frame).to_csv(args.out / "test_paired_effects.csv", index=False)
    print(json.dumps({"out": str(args.out), "states": n_states, "fit_end": fit_end, "validation_end": val_end, "shrinkage": shrinkage}, indent=2))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
