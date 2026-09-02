#!/usr/bin/env python3
"""Reproduce the PEMS-BAY Wasserstein analysis from processed Gaussian states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from bwar.gaussian_geometry import bw2_cov  # noqa: E402
from bwar.paper_jcgs.gaussian_models import (  # noqa: E402
    fit_var,
    recursive_predict_z,
)
from run_pems_bay import (  # noqa: E402
    HORIZONS,
    SEED,
    _encode_methods,
    _method_intervals,
    _paired_effects,
    _select_ridge,
)


def _load_states(data_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    means_frame = pd.read_csv(data_dir / "pems_bay_gaussian_means.csv")
    covariances_frame = pd.read_csv(
        data_dir / "pems_bay_gaussian_covariances.csv"
    )
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for panel in sorted(means_frame["panel"].unique()):
        one_mean = means_frame[means_frame["panel"] == panel]
        n_states = int(one_mean["state_index"].max()) + 1
        dimension = int(one_mean["sensor_rank"].max())
        means = np.empty((n_states, dimension), dtype=float)
        for row in one_mean.itertuples(index=False):
            means[int(row.state_index), int(row.sensor_rank) - 1] = float(
                row.standardized_mean
            )

        one_covariance = covariances_frame[
            covariances_frame["panel"] == panel
        ]
        covariances = np.zeros((n_states, dimension, dimension), dtype=float)
        for row in one_covariance.itertuples(index=False):
            state = int(row.state_index)
            row_index = int(row.row_sensor_rank) - 1
            column_index = int(row.column_sensor_rank) - 1
            value = float(row.standardized_covariance)
            covariances[state, row_index, column_index] = value
            covariances[state, column_index, row_index] = value
        output[int(panel)] = means, covariances
    return output


def _record(
    *,
    panel: int,
    source: int,
    horizon: int,
    method: str,
    prediction: tuple[np.ndarray, np.ndarray],
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
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
    }


def _evaluate_panel(
    panel: int,
    means: np.ndarray,
    covariances: np.ndarray,
    fit_end: int,
    validation_end: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    encodings = _encode_methods(means, covariances, fit_end)
    selected: list[dict[str, float | int | str]] = []
    matrices: dict[tuple[str, int], tuple[np.ndarray, object]] = {}
    for name, (coordinates, decoder) in encodings.items():
        for horizon in HORIZONS:
            ridge = _select_ridge(
                means,
                covariances,
                coordinates,
                decoder,
                fit_end,
                validation_end,
                horizon,
            )
            matrices[(name, horizon)] = (
                fit_var(coordinates, validation_end, lam=ridge, model="diag"),
                decoder,
            )
            selected.append(
                {
                    "panel": panel,
                    "method": name,
                    "horizon": horizon,
                    "ridge": ridge,
                }
            )

    records: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        for source in range(validation_end, len(means) - horizon):
            target = source + horizon
            for name, (coordinates, _) in encodings.items():
                matrix, decoder = matrices[(name, horizon)]
                prediction = decoder(
                    recursive_predict_z(coordinates[source], matrix, horizon)
                )
                records.append(
                    _record(
                        panel=panel,
                        source=source,
                        horizon=horizon,
                        method=name,
                        prediction=prediction,
                        target_mean=means[target],
                        target_covariance=covariances[target],
                    )
                )
            records.append(
                _record(
                    panel=panel,
                    source=source,
                    horizon=horizon,
                    method="Persistence",
                    prediction=(means[source], covariances[source]),
                    target_mean=means[target],
                    target_covariance=covariances[target],
                )
            )
    return records, selected


def _write_summary(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    summary = records.groupby(["method", "horizon"], as_index=False).agg(
        w2_squared_mean=("w2_squared", "mean"),
        w2_squared_median=("w2_squared", "median"),
        covariance_w2_component_mean=("covariance_w2_component", "mean"),
        mean_rmse_mean=("mean_rmse", "mean"),
        n=("w2_squared", "size"),
    )
    intervals = _method_intervals(records)
    for metric in ("w2_squared", "covariance_w2_component", "mean_rmse"):
        subset = intervals[intervals.metric == metric][
            ["method", "horizon", "standard_error", "ci_low", "ci_high"]
        ].rename(
            columns={
                "standard_error": f"{metric}_standard_error",
                "ci_low": f"{metric}_ci_low",
                "ci_high": f"{metric}_ci_high",
            }
        )
        summary = summary.merge(
            subset,
            on=["method", "horizon"],
            how="left",
            validate="one_to_one",
        )
    summary.to_csv(path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/pems_bay_from_states")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.data_dir / "analysis_protocol.json").read_text())
    fit_end = int(protocol["fit_end"])
    validation_end = int(protocol["validation_end"])
    np.random.seed(SEED)

    states = _load_states(args.data_dir)
    all_records: list[dict[str, float | int | str]] = []
    all_ridges: list[dict[str, float | int | str]] = []
    for panel, (means, covariances) in states.items():
        records, ridges = _evaluate_panel(
            panel, means, covariances, fit_end, validation_end
        )
        all_records.extend(records)
        all_ridges.extend(ridges)

    record_frame = pd.DataFrame(all_records)
    record_frame.to_csv(args.out / "test_origin_level.csv", index=False)
    pd.DataFrame(all_ridges).to_csv(
        args.out / "validation_selected_ridges.csv", index=False
    )
    summary = _write_summary(record_frame, args.out / "test_method_summary.csv")
    _paired_effects(record_frame).to_csv(
        args.out / "test_paired_effects.csv", index=False
    )
    print(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "out": str(args.out),
                "panels": len(states),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
