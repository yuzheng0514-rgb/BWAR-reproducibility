#!/usr/bin/env python3
"""Check completeness, continuity, and numerical integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_METHODS = (
    "persistence",
    "euclidean",
    "cholesky",
    "log_euclidean",
    "fixed",
    "local",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.result_dir / "protocol.json").read_text())
    repository_root = Path(__file__).resolve().parents[1]
    config_payload = json.loads(
        (repository_root / protocol["config"]).read_text(encoding="utf-8")
    )
    config = config_payload["design"]
    raw = pd.read_csv(args.result_dir / "replication_results.csv.gz")
    primary = raw.loc[raw.method.isin(PRIMARY_METHODS)].copy()
    expected_reps = int(protocol["replications"])
    expected_controls = {
        "persistent": tuple(float(v) for v in config["target_deltas"]),
        "mean_reverting": tuple(float(v) for v in config["mean_reversion_controls"]),
    }
    expected_rows = (
        expected_reps
        * sum(len(values) for values in expected_controls.values())
        * len(config["horizons"])
        * len(PRIMARY_METHODS)
    )
    group_counts = primary.groupby(
        ["regime", "movement_control", "horizon", "method"]
    ).replication.nunique()
    control_match = all(
        tuple(sorted(primary.loc[primary.regime.eq(regime), "movement_control"].unique()))
        == controls
        for regime, controls in expected_controls.items()
    )
    numeric_columns = (
        "test_w2_mean",
        "average_step_movement",
        "minimum_step_movement",
        "fit_boundary_step_to_median_ratio",
        "validation_boundary_step_to_median_ratio",
        "minimum_raw_generating_transport_eigenvalue",
    )
    finite = bool(np.isfinite(primary.loc[:, numeric_columns].to_numpy(float)).all())
    sequence_counts = primary.groupby(
        ["replication", "regime", "movement_control"]
    ).sequence_sha256.nunique()
    checks = {
        "expected_primary_row_count": len(primary) == expected_rows,
        "complete_replications_per_cell": bool((group_counts == expected_reps).all()),
        "complete_row_specific_control_grids": bool(control_match),
        "all_primary_numeric_outputs_finite": finite,
        "one_sequence_hash_per_case": bool((sequence_counts == 1).all()),
        "covariance_changes_at_every_adjacent_time": bool(
            (primary.minimum_step_movement > 0.0).all()
        ),
        "no_large_fit_boundary_step": bool(
            (primary.fit_boundary_step_to_median_ratio < 2.0).all()
        ),
        "no_large_validation_test_boundary_step": bool(
            (primary.validation_boundary_step_to_median_ratio < 2.0).all()
        ),
        "no_generator_transport_clipping": bool(
            (primary.generating_transport_clip_count == 0).all()
        ),
        "generator_transport_above_configured_floor": bool(
            (
                primary.minimum_raw_generating_transport_eigenvalue
                > float(config["minimum_generating_transport_eigenvalue"])
            ).all()
        ),
        "no_fixed_or_local_prediction_repairs": bool(
            (
                primary.loc[
                    primary.method.isin(("fixed", "local")),
                    "prediction_repair_count",
                ]
                == 0
            ).all()
        ),
        "ridge_selected_only_from_configured_grid": bool(
            primary.loc[primary.method.ne("persistence"), "selected_ridge"].isin(
                config["ridge_grid"]
            ).all()
        ),
    }
    mean_loss = (
        primary.groupby(
            ["regime", "movement_control", "horizon", "method"],
            as_index=False,
        )
        .test_w2_mean.mean()
    )
    winners = mean_loss.loc[
        mean_loss.groupby(["regime", "movement_control", "horizon"]).test_w2_mean.idxmin()
    ].sort_values(["regime", "movement_control", "horizon"])
    diagnostics = {
        "primary_rows_observed": int(len(primary)),
        "primary_rows_expected": int(expected_rows),
        "minimum_observed_covariance_step": float(primary.minimum_step_movement.min()),
        "maximum_fit_boundary_to_median_ratio": float(
            primary.fit_boundary_step_to_median_ratio.max()
        ),
        "maximum_validation_boundary_to_median_ratio": float(
            primary.validation_boundary_step_to_median_ratio.max()
        ),
        "minimum_raw_generating_transport_eigenvalue": float(
            primary.minimum_raw_generating_transport_eigenvalue.min()
        ),
        "prediction_repairs_by_method": {
            str(key): int(value)
            for key, value in primary.groupby("method").prediction_repair_count.sum().items()
        },
        "winners": winners.to_dict(orient="records"),
    }
    report = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": diagnostics,
    }
    (args.result_dir / "integrity_checks.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    winners.to_csv(args.result_dir / "winner_table.csv", index=False)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
