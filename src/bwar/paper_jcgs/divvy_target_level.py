from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from bwar.paper_jcgs.local_reference_bwar import (
    build_local_bwar_geometry,
    forecast_local_bwar,
)
import bwar.paper_jcgs.divvy_artifacts as realdata_artifacts
import bwar.paper_jcgs.rolling_origin as rolling
from bwar.paper_jcgs.gaussian_models import (
    domain_loss_from_moments,
    gaussian_w2_squared,
)


TARGET_COLUMNS = (
    "origin",
    "fit_end",
    "val_end",
    "test_end",
    "horizon",
    "source_index",
    "target_index",
    "method",
    "window_length",
    "ridge",
    "raw_rmse",
    "w2",
    "min_pred_eig",
    "reference_residual",
    "reference_refreshed",
    "reference_fallback",
)

INFERENCE_COLUMNS = (
    "target_method",
    "comparator",
    "horizon",
    "metric",
    "mean_difference",
    "ci_low",
    "ci_high",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
    "p_value",
    "p_holm",
    "n_targets",
    "n_origins",
    "block_length",
    "sensitivity_block_length",
    "replicates",
    "seed",
)

METHOD_SUMMARY_COLUMNS = (
    "method",
    "horizon",
    "metric",
    "mean",
    "ci_low",
    "ci_high",
    "n_targets",
    "n_origins",
    "block_length",
    "replicates",
    "seed",
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def test_source_indices(val_end: int, test_end: int, *, horizon: int) -> np.ndarray:
    validation_end = _positive_integer(val_end, name="val_end")
    testing_end = _positive_integer(test_end, name="test_end")
    forecast_horizon = _positive_integer(horizon, name="horizon")
    if forecast_horizon > validation_end or validation_end >= testing_end:
        raise ValueError("invalid validation/test boundaries or horizon")
    return np.arange(
        validation_end - forecast_horizon,
        testing_end - forecast_horizon,
        dtype=int,
    )


def validation_source_indices(
    fit_end: int,
    val_end: int,
    *,
    horizon: int,
    max_window_length: int,
) -> np.ndarray:
    fitting_end = _positive_integer(fit_end, name="fit_end")
    validation_end = _positive_integer(val_end, name="val_end")
    forecast_horizon = _positive_integer(horizon, name="horizon")
    maximum_window = _positive_integer(max_window_length, name="max_window_length")
    if fitting_end >= validation_end or forecast_horizon > fitting_end:
        raise ValueError("invalid fitting/validation boundaries or horizon")
    start = max(fitting_end - forecast_horizon, maximum_window - 1)
    stop = validation_end - forecast_horizon
    if stop <= start:
        raise ValueError("validation block has no source with the requested window")
    return np.arange(start, stop, dtype=int)


def validate_complete_panel(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("complete paired panel must be a nonempty DataFrame")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("complete paired panel methods must be unique and nonempty")
    required = {
        "origin",
        "horizon",
        "target_index",
        "method",
        "raw_rmse",
        "w2",
    }
    if not required.issubset(frame.columns):
        raise ValueError("complete paired panel is missing required columns")
    keys = ["origin", "horizon", "target_index"]
    if frame.duplicated(keys + ["method"]).any():
        raise ValueError("complete paired panel contains duplicate method-target rows")
    if frame.duplicated(["horizon", "target_index", "method"]).any():
        raise ValueError("complete paired panel contains a target in multiple origins")
    observed_methods = set(frame["method"].astype(str))
    if observed_methods != set(methods):
        raise ValueError("complete paired panel does not contain exactly the requested methods")
    grouped = frame.groupby(["origin", "horizon", "method"], sort=True)[
        "target_index"
    ].apply(lambda values: tuple(sorted(int(value) for value in values)))
    for origin in sorted(frame["origin"].unique()):
        for horizon in sorted(frame["horizon"].unique()):
            target_sets = [grouped.get((origin, horizon, method), ()) for method in methods]
            if any(not targets for targets in target_sets) or any(
                targets != target_sets[0] for targets in target_sets[1:]
            ):
                raise ValueError("complete paired panel has unequal target indices")
    losses = frame[["raw_rmse", "w2"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(losses.to_numpy(dtype=float)).all():
        raise ValueError("complete paired panel contains nonfinite losses")
    return frame.sort_values(keys + ["method"]).reset_index(drop=True)


def moving_block_mean_bootstrap(
    by_origin: dict[int, np.ndarray],
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    block_size = _positive_integer(block_length, name="block_length")
    n_replicates = _positive_integer(replicates, name="replicates")
    if not isinstance(by_origin, dict) or not by_origin:
        raise ValueError("by_origin must be a nonempty dictionary")
    normalized: dict[int, np.ndarray] = {}
    for origin, raw_values in by_origin.items():
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("each origin must contain finite one-dimensional values")
        normalized[int(origin)] = values
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_replicates, dtype=float)
    for draw_index in range(n_replicates):
        sampled_origins: list[np.ndarray] = []
        for origin in sorted(normalized):
            values = normalized[origin]
            effective_block_size = min(block_size, len(values))
            starts = np.arange(len(values) - effective_block_size + 1, dtype=int)
            blocks: list[np.ndarray] = []
            sampled_length = 0
            while sampled_length < len(values):
                start = int(rng.choice(starts))
                block = values[start : start + effective_block_size]
                blocks.append(block)
                sampled_length += len(block)
            sampled_origins.append(np.concatenate(blocks)[: len(values)])
        draws[draw_index] = float(np.concatenate(sampled_origins).mean())
    return draws


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a nonempty one-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values between zero and one")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    scaled = np.minimum(1.0, (len(values) - np.arange(len(values))) * sorted_values)
    monotone = np.maximum.accumulate(scaled)
    adjusted = np.empty_like(monotone)
    adjusted[order] = monotone
    return adjusted


def _differences_by_origin(
    panel: pd.DataFrame,
    *,
    target_method: str,
    comparator: str,
    metric: str,
) -> dict[int, np.ndarray]:
    indexed = panel.pivot(
        index=["origin", "target_index"], columns="method", values=metric
    )
    differences = indexed[target_method] - indexed[comparator]
    return {
        int(origin): group.to_numpy(dtype=float)
        for origin, group in differences.groupby(level="origin", sort=True)
    }


def paired_inference(
    panel: pd.DataFrame,
    *,
    targets: tuple[str, ...] = ("fixed_bwar", "local_bwar"),
    comparators: tuple[str, ...],
    horizon: int,
    block_length: int = 3,
    sensitivity_block_length: int = 6,
    replicates: int = 10_000,
    seed: int = 20_260_714,
) -> pd.DataFrame:
    forecast_horizon = _positive_integer(horizon, name="horizon")
    if not targets or not comparators or set(targets) & set(comparators):
        raise ValueError("targets and comparators must be nonempty and disjoint")
    requested_methods = tuple(targets) + tuple(comparators)
    selected = panel.loc[
        panel["horizon"].eq(forecast_horizon)
        & panel["method"].isin(requested_methods)
    ].copy()
    selected = validate_complete_panel(selected, requested_methods)
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for target_method in targets:
        for metric in ("raw_rmse", "w2"):
            group_start = len(rows)
            for comparator in comparators:
                differences = _differences_by_origin(
                    selected,
                    target_method=target_method,
                    comparator=comparator,
                    metric=metric,
                )
                values = np.concatenate([differences[key] for key in sorted(differences)])
                observed = float(values.mean())
                comparison_seed = int(seed) + comparison_index * 1009
                bootstrap = moving_block_mean_bootstrap(
                    differences,
                    block_length=block_length,
                    replicates=replicates,
                    seed=comparison_seed,
                )
                sensitivity = moving_block_mean_bootstrap(
                    differences,
                    block_length=sensitivity_block_length,
                    replicates=replicates,
                    seed=comparison_seed + 503,
                )
                centered = {
                    origin: origin_values - observed
                    for origin, origin_values in differences.items()
                }
                null_bootstrap = moving_block_mean_bootstrap(
                    centered,
                    block_length=block_length,
                    replicates=replicates,
                    seed=comparison_seed + 997,
                )
                p_value = float(
                    (1 + np.count_nonzero(null_bootstrap <= observed))
                    / (len(null_bootstrap) + 1)
                )
                rows.append(
                    {
                        "target_method": target_method,
                        "comparator": comparator,
                        "horizon": forecast_horizon,
                        "metric": metric,
                        "mean_difference": observed,
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "sensitivity_ci_low": float(np.quantile(sensitivity, 0.025)),
                        "sensitivity_ci_high": float(np.quantile(sensitivity, 0.975)),
                        "p_value": p_value,
                        "p_holm": np.nan,
                        "n_targets": int(len(values)),
                        "n_origins": int(len(differences)),
                        "block_length": int(block_length),
                        "sensitivity_block_length": int(sensitivity_block_length),
                        "replicates": int(replicates),
                        "seed": comparison_seed,
                    }
                )
                comparison_index += 1
            group_stop = len(rows)
            adjusted = holm_adjust(
                np.asarray(
                    [rows[index]["p_value"] for index in range(group_start, group_stop)],
                    dtype=float,
                )
            )
            for index, p_holm in zip(range(group_start, group_stop), adjusted, strict=True):
                rows[index]["p_holm"] = float(p_holm)
    return pd.DataFrame(rows, columns=INFERENCE_COLUMNS)


def method_level_summary(
    panel: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    block_length: int = 3,
    replicates: int = 10_000,
    seed: int = 20_260_714,
) -> pd.DataFrame:
    """Summarize target-weighted losses with origin-preserving block intervals."""
    selected = panel.loc[panel["method"].isin(methods)].copy()
    selected = validate_complete_panel(selected, methods)
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for horizon in sorted(int(value) for value in selected["horizon"].unique()):
        horizon_panel = selected.loc[selected["horizon"].eq(horizon)]
        for metric in ("raw_rmse", "w2"):
            for method in methods:
                method_panel = horizon_panel.loc[horizon_panel["method"].eq(method)]
                by_origin = {
                    int(origin): group.sort_values("target_index")[metric].to_numpy(
                        dtype=float
                    )
                    for origin, group in method_panel.groupby("origin", sort=True)
                }
                values = np.concatenate(
                    [by_origin[origin] for origin in sorted(by_origin)]
                )
                comparison_seed = int(seed) + comparison_index * 1009
                bootstrap = moving_block_mean_bootstrap(
                    by_origin,
                    block_length=block_length,
                    replicates=replicates,
                    seed=comparison_seed,
                )
                rows.append(
                    {
                        "method": method,
                        "horizon": horizon,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "n_targets": int(len(values)),
                        "n_origins": int(len(by_origin)),
                        "block_length": int(block_length),
                        "replicates": int(replicates),
                        "seed": comparison_seed,
                    }
                )
                comparison_index += 1
    return pd.DataFrame(rows, columns=METHOD_SUMMARY_COLUMNS)


def _validated_gaussian_series(
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(means, dtype=float)
    covariance_array = np.asarray(covariances, dtype=float)
    if mean_array.ndim != 2 or mean_array.shape[1] == 0:
        raise ValueError("means must be a positive-dimension two-dimensional array")
    if covariance_array.shape != (
        len(mean_array),
        mean_array.shape[1],
        mean_array.shape[1],
    ):
        raise ValueError("covariances must match the Gaussian mean series")
    if not np.isfinite(mean_array).all() or not np.isfinite(covariance_array).all():
        raise ValueError("Gaussian series must be finite")
    return mean_array, covariance_array


def _evaluate_local_sources(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    sources: np.ndarray,
    origin: int,
    fit_end: int,
    val_end: int,
    test_end: int,
    horizon: int,
    window_length: int,
    ridge: float,
    domain_profile: Mapping[str, object],
) -> pd.DataFrame:
    geometries = build_local_bwar_geometry(
        means,
        covariances,
        window_length=window_length,
        source_indices=sources,
        refresh_period=1,
    )
    rows: list[dict[str, object]] = []
    for source in sources:
        source_index = int(source)
        geometry = geometries[source_index]
        predicted_mean, predicted_covariance = forecast_local_bwar(
            geometry,
            ridge=float(ridge),
            horizon=int(horizon),
        )
        target_index = source_index + int(horizon)
        rows.append(
            target_loss_row(
                origin=origin,
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                horizon=horizon,
                source_index=source_index,
                method="local_bwar",
                window_length=window_length,
                ridge=float(ridge),
                pred_mean=predicted_mean,
                pred_cov=predicted_covariance,
                target_mean=means[target_index],
                target_cov=covariances[target_index],
                domain_profile=domain_profile,
                reference_residual=float(geometry.reference.residual),
                reference_refreshed=bool(geometry.reference.refreshed),
                reference_fallback=bool(geometry.reference.fallback),
            )
        )
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def evaluate_local_split(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    origin: int,
    fit_end: int,
    val_end: int,
    test_end: int,
    horizon: int,
    window_grid: tuple[int, ...] = (24, 48, 72),
    ridge_grid: tuple[float, ...],
    domain_profile: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mean_array, covariance_array = _validated_gaussian_series(means, covariances)
    if not window_grid or len(set(window_grid)) != len(window_grid):
        raise ValueError("window_grid must contain unique positive window lengths")
    windows = tuple(_positive_integer(value, name="window_length") for value in window_grid)
    if min(windows) < 3:
        raise ValueError("local window lengths must be at least three")
    ridges = tuple(float(value) for value in ridge_grid)
    if not ridges or not np.isfinite(ridges).all() or any(value <= 0.0 for value in ridges):
        raise ValueError("ridge_grid must contain finite positive values")
    validation_sources = validation_source_indices(
        fit_end,
        val_end,
        horizon=horizon,
        max_window_length=max(windows),
    )
    tuning_rows: list[dict[str, object]] = []
    for window_length in windows:
        geometries = build_local_bwar_geometry(
            mean_array,
            covariance_array,
            window_length=window_length,
            source_indices=validation_sources,
            refresh_period=1,
        )
        for ridge in ridges:
            losses: list[dict[str, object]] = []
            for source in validation_sources:
                source_index = int(source)
                predicted_mean, predicted_covariance = forecast_local_bwar(
                    geometries[source_index],
                    ridge=ridge,
                    horizon=horizon,
                )
                target_index = source_index + int(horizon)
                losses.append(
                    target_loss_row(
                        origin=origin,
                        fit_end=fit_end,
                        val_end=val_end,
                        test_end=test_end,
                        horizon=horizon,
                        source_index=source_index,
                        method="local_bwar",
                        window_length=window_length,
                        ridge=ridge,
                        pred_mean=predicted_mean,
                        pred_cov=predicted_covariance,
                        target_mean=mean_array[target_index],
                        target_cov=covariance_array[target_index],
                        domain_profile=domain_profile,
                    )
                )
            loss_frame = pd.DataFrame(losses)
            tuning_rows.append(
                {
                    "origin": int(origin),
                    "horizon": int(horizon),
                    "window_length": int(window_length),
                    "ridge": float(ridge),
                    "validation_raw_rmse": float(loss_frame["raw_rmse"].mean()),
                    "validation_w2": float(loss_frame["w2"].mean()),
                    "n_validation_targets": int(len(loss_frame)),
                    "validation_target_start": int(
                        validation_sources[0] + int(horizon)
                    ),
                    "validation_target_stop": int(
                        validation_sources[-1] + int(horizon) + 1
                    ),
                }
            )
    tuning = pd.DataFrame(tuning_rows)
    best = min(
        tuning_rows,
        key=lambda row: (
            float(row["validation_raw_rmse"]),
            int(row["window_length"]),
            -float(row["ridge"]),
        ),
    )
    selected = {
        "origin": int(origin),
        "horizon": int(horizon),
        "window_length": int(best["window_length"]),
        "ridge": float(best["ridge"]),
        "validation_raw_rmse": float(best["validation_raw_rmse"]),
        "validation_w2": float(best["validation_w2"]),
        "n_validation_targets": int(best["n_validation_targets"]),
    }
    test_sources = test_source_indices(val_end, test_end, horizon=horizon)
    test_rows = _evaluate_local_sources(
        mean_array,
        covariance_array,
        sources=test_sources,
        origin=origin,
        fit_end=fit_end,
        val_end=val_end,
        test_end=test_end,
        horizon=horizon,
        window_length=int(selected["window_length"]),
        ridge=float(selected["ridge"]),
        domain_profile=domain_profile,
    )
    return test_rows, tuning, selected


def _append_target_forecasts(
    rows: list[dict[str, object]],
    *,
    means: np.ndarray,
    covariances: np.ndarray,
    sources: np.ndarray,
    forecast,
    origin: int,
    fit_end: int,
    val_end: int,
    test_end: int,
    horizon: int,
    method: str,
    window_length: float | int,
    ridge: float,
    domain_profile: Mapping[str, object],
) -> None:
    for source in sources:
        source_index = int(source)
        predicted_mean, predicted_covariance = forecast(source_index)
        target_index = source_index + int(horizon)
        rows.append(
            target_loss_row(
                origin=origin,
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                horizon=horizon,
                source_index=source_index,
                method=method,
                window_length=window_length,
                ridge=ridge,
                pred_mean=predicted_mean,
                pred_cov=predicted_covariance,
                target_mean=means[target_index],
                target_cov=covariances[target_index],
                domain_profile=domain_profile,
            )
        )


def _selected_raw_var(
    means: np.ndarray,
    covariances: np.ndarray,
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    window_size: int,
    ridge_grid: tuple[float, ...],
    domain_profile: Mapping[str, object],
) -> tuple[np.ndarray, float]:
    fit_raw_end = int(window_starts[fit_end - 1] + window_size)
    validation_raw_end = int(window_starts[val_end - 1] + window_size)
    validation_start = max(0, fit_end - horizon)
    validation_stop = max(validation_start, val_end - horizon)
    candidates: list[tuple[float, float]] = []
    for ridge in ridge_grid:
        weights = rolling._fit_raw_var(
            raw_series,
            fit_raw_end,
            lam=float(ridge),
            model="full",
        )
        metrics = rolling._score_raw_var_with_weights(
            means,
            covariances,
            raw_series,
            window_starts,
            window_size=window_size,
            W=weights,
            start_t=validation_start,
            stop_t=validation_stop,
            horizon=horizon,
            domain_profile=dict(domain_profile),
        )
        score = float(metrics["domain_loss_mean"])
        if np.isfinite(score):
            candidates.append((score, float(ridge)))
    if not candidates:
        raise ValueError("raw VAR has no finite validation candidate")
    _, selected_ridge = min(candidates, key=lambda item: (item[0], -item[1]))
    final_weights = rolling._fit_raw_var(
        raw_series,
        validation_raw_end,
        lam=selected_ridge,
        model="full",
    )
    return final_weights, selected_ridge


def evaluate_fixed_split(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
    origin: int,
    fit_end: int,
    val_end: int,
    test_end: int,
    horizon: int,
    domain_profile: Mapping[str, object],
    ridge_grid: tuple[float, ...],
) -> pd.DataFrame:
    mean_array, covariance_array = _validated_gaussian_series(means, covariances)
    raw_array = np.asarray(raw_series, dtype=float)
    starts = np.asarray(window_starts, dtype=int)
    if len(starts) != len(mean_array):
        raise ValueError("window starts must match the Gaussian series")
    if raw_array.ndim != 2 or raw_array.shape[1] != mean_array.shape[1]:
        raise ValueError("raw_series must match the Gaussian dimension")
    sources = test_source_indices(val_end, test_end, horizon=horizon)
    rows: list[dict[str, object]] = []

    _append_target_forecasts(
        rows,
        means=mean_array,
        covariances=covariance_array,
        sources=sources,
        forecast=lambda source: (mean_array[source], covariance_array[source]),
        origin=origin,
        fit_end=fit_end,
        val_end=val_end,
        test_end=test_end,
        horizon=horizon,
        method="persistence",
        window_length=np.nan,
        ridge=np.nan,
        domain_profile=domain_profile,
    )

    raw_weights, raw_ridge = _selected_raw_var(
        mean_array,
        covariance_array,
        raw_array,
        starts,
        fit_end=fit_end,
        val_end=val_end,
        horizon=horizon,
        window_size=window_size,
        ridge_grid=ridge_grid,
        domain_profile=domain_profile,
    )

    def raw_forecast(source: int) -> tuple[np.ndarray, np.ndarray]:
        predicted_window = rolling._predict_raw_target_window(
            raw_array,
            starts,
            source_index=source,
            horizon=horizon,
            window_size=window_size,
            W=raw_weights,
        )
        return rolling._window_moments(predicted_window)

    _append_target_forecasts(
        rows,
        means=mean_array,
        covariances=covariance_array,
        sources=sources,
        forecast=raw_forecast,
        origin=origin,
        fit_end=fit_end,
        val_end=val_end,
        test_end=test_end,
        horizon=horizon,
        method="raw_var_window_ar",
        window_length=window_size,
        ridge=raw_ridge,
        domain_profile=domain_profile,
    )

    dimension = mean_array.shape[1]
    encoded_specs = (
        (
            "euclidean_gaussian_ar",
            rolling.euclidean_encode,
            lambda coordinate: rolling.euclidean_decode(coordinate, dimension),
        ),
        (
            "cholesky_gaussian_ar",
            rolling.cholesky_encode,
            lambda coordinate: rolling.cholesky_decode(coordinate, dimension),
        ),
        (
            "log_euclidean_gaussian_ar",
            rolling.log_euclidean_encode,
            lambda coordinate: rolling.log_euclidean_decode(coordinate, dimension),
        ),
    )
    for method, encode, decode in encoded_specs:
        selected = rolling.select_encoded_model(
            mean_array,
            covariance_array,
            encode,
            decode,
            fit_end=fit_end,
            val_end=val_end,
            horizon=horizon,
            ar_model="full",
            selection_metric="domain",
            ridge_grid=ridge_grid,
            domain_profile=dict(domain_profile),
        )

        def encoded_forecast(
            source: int,
            selected_model: dict[str, object] = selected,
        ) -> tuple[np.ndarray, np.ndarray]:
            coordinate = rolling.recursive_predict_z(
                selected_model["Z"][source],
                selected_model["W"],
                horizon,
            )
            return selected_model["decode"](coordinate)

        _append_target_forecasts(
            rows,
            means=mean_array,
            covariances=covariance_array,
            sources=sources,
            forecast=encoded_forecast,
            origin=origin,
            fit_end=fit_end,
            val_end=val_end,
            test_end=test_end,
            horizon=horizon,
            method=method,
            window_length=val_end,
            ridge=float(selected["ridge"]),
            domain_profile=domain_profile,
        )

    _, reference_mean, reference_covariance = realdata_artifacts.barycenter_only_references(
        mean_array[:fit_end], covariance_array[:fit_end]
    )[0]
    fixed_selected = rolling.select_encoded_model(
        mean_array,
        covariance_array,
        lambda mean, covariance: rolling.bwar_gaussian_encode(
            mean,
            covariance,
            reference_mean,
            reference_covariance,
        ),
        lambda coordinate: rolling.bwar_gaussian_decode(
            coordinate,
            reference_mean,
            reference_covariance,
        ),
        fit_end=fit_end,
        val_end=val_end,
        horizon=horizon,
        ar_model="full",
        selection_metric="domain",
        ridge_grid=ridge_grid,
        domain_profile=dict(domain_profile),
    )

    def fixed_bwar_forecast(source: int) -> tuple[np.ndarray, np.ndarray]:
        coordinate = rolling.recursive_predict_z(
            fixed_selected["Z"][source],
            fixed_selected["W"],
            horizon,
        )
        return fixed_selected["decode"](coordinate)

    _append_target_forecasts(
        rows,
        means=mean_array,
        covariances=covariance_array,
        sources=sources,
        forecast=fixed_bwar_forecast,
        origin=origin,
        fit_end=fit_end,
        val_end=val_end,
        test_end=test_end,
        horizon=horizon,
        method="fixed_bwar",
        window_length=val_end,
        ridge=float(fixed_selected["ridge"]),
        domain_profile=domain_profile,
    )
    methods = (
        "persistence",
        "raw_var_window_ar",
        "euclidean_gaussian_ar",
        "cholesky_gaussian_ar",
        "log_euclidean_gaussian_ar",
        "fixed_bwar",
    )
    return validate_complete_panel(pd.DataFrame(rows, columns=TARGET_COLUMNS), methods)


def evaluate_target_panel(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
    splits: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    horizons: tuple[int, ...],
    domain_profile: Mapping[str, object],
    ridge_grid: tuple[float, ...],
    local_window_grid: tuple[int, ...] = (24, 48, 72),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_array, covariance_array = _validated_gaussian_series(means, covariances)
    split_values = tuple(tuple(int(value) for value in split) for split in splits)
    if not split_values or any(
        len(split) != 3
        or not (2 <= split[0] < split[1] < split[2] <= len(mean_array))
        for split in split_values
    ):
        raise ValueError("splits must contain valid fit-validation-test boundaries")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain unique positive values")
    horizon_values = tuple(_positive_integer(value, name="horizon") for value in horizons)
    frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    for origin, (fit_end, val_end, test_end) in enumerate(split_values):
        for horizon in horizon_values:
            if test_end <= val_end + horizon:
                continue
            fixed = evaluate_fixed_split(
                mean_array,
                covariance_array,
                raw_series=raw_series,
                window_starts=window_starts,
                window_size=window_size,
                origin=origin,
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                horizon=horizon,
                domain_profile=domain_profile,
                ridge_grid=ridge_grid,
            )
            local, tuning, selected = evaluate_local_split(
                mean_array,
                covariance_array,
                origin=origin,
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                horizon=horizon,
                window_grid=local_window_grid,
                ridge_grid=ridge_grid,
                domain_profile=domain_profile,
            )
            combined = pd.concat([fixed, local], ignore_index=True)
            methods = (
                "persistence",
                "raw_var_window_ar",
                "euclidean_gaussian_ar",
                "cholesky_gaussian_ar",
                "log_euclidean_gaussian_ar",
                "fixed_bwar",
                "local_bwar",
            )
            frames.append(validate_complete_panel(combined, methods))
            tuning_frames.append(tuning)
            selected_rows.append(selected)
    if not frames:
        raise ValueError("no split-horizon combination produced target forecasts")
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["origin", "horizon", "target_index", "method"]).any():
        raise ValueError("target panel contains duplicate forecast rows")
    return (
        panel.sort_values(
            ["origin", "horizon", "target_index", "method"]
        ).reset_index(drop=True),
        pd.concat(tuning_frames, ignore_index=True),
        pd.DataFrame(selected_rows),
    )


def target_loss_row(
    *,
    origin: int,
    fit_end: int,
    val_end: int,
    test_end: int,
    horizon: int,
    source_index: int,
    method: str,
    window_length: float | int,
    ridge: float,
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    domain_profile: Mapping[str, object],
    reference_residual: float = np.nan,
    reference_refreshed: object = np.nan,
    reference_fallback: object = np.nan,
) -> dict[str, object]:
    predicted_mean = np.asarray(pred_mean, dtype=float)
    predicted_covariance = np.asarray(pred_cov, dtype=float)
    observed_mean = np.asarray(target_mean, dtype=float)
    observed_covariance = np.asarray(target_cov, dtype=float)
    if predicted_mean.ndim != 1 or observed_mean.shape != predicted_mean.shape:
        raise ValueError("predicted and target means must have the same vector shape")
    dimension = len(predicted_mean)
    expected_covariance_shape = (dimension, dimension)
    if (
        predicted_covariance.shape != expected_covariance_shape
        or observed_covariance.shape != expected_covariance_shape
        or not np.isfinite(predicted_mean).all()
        or not np.isfinite(predicted_covariance).all()
        or not np.isfinite(observed_mean).all()
        or not np.isfinite(observed_covariance).all()
        or not np.allclose(predicted_covariance, predicted_covariance.T)
    ):
        raise ValueError("predicted moments must be finite SPD and match the target")
    try:
        eigenvalues = np.linalg.eigvalsh(predicted_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("predicted covariance must be finite SPD") from exc
    minimum_eigenvalue = float(eigenvalues.min())
    if not np.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 0.0:
        raise ValueError("predicted covariance must be finite SPD")
    forecast_horizon = _positive_integer(horizon, name="horizon")
    source = int(source_index)
    raw_rmse = float(
        domain_loss_from_moments(
            predicted_mean,
            predicted_covariance,
            observed_mean,
            observed_covariance,
            dict(domain_profile),
        )
    )
    w2 = float(
        gaussian_w2_squared(
            predicted_mean,
            predicted_covariance,
            observed_mean,
            observed_covariance,
        )
    )
    if not np.isfinite(raw_rmse) or not np.isfinite(w2):
        raise ValueError("target losses must be finite")
    return {
        "origin": int(origin),
        "fit_end": int(fit_end),
        "val_end": int(val_end),
        "test_end": int(test_end),
        "horizon": forecast_horizon,
        "source_index": source,
        "target_index": source + forecast_horizon,
        "method": str(method),
        "window_length": window_length,
        "ridge": float(ridge),
        "raw_rmse": raw_rmse,
        "w2": w2,
        "min_pred_eig": minimum_eigenvalue,
        "reference_residual": reference_residual,
        "reference_refreshed": reference_refreshed,
        "reference_fallback": reference_fallback,
    }
