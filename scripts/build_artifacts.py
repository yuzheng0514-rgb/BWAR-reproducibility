#!/usr/bin/env python3
"""Build the manuscript tables and S2 figure from compact result files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DIVVY_METHODS = (
    "persistence",
    "raw_var_window_ar",
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "fixed_bwar",
    "local_bwar",
)
DIVVY_LABELS = {
    "persistence": "Persistence",
    "raw_var_window_ar": "Raw VAR",
    "euclidean_gaussian_ar": "Euclidean AR",
    "cholesky_gaussian_ar": "Cholesky AR",
    "log_euclidean_gaussian_ar": "log-Euclidean AR",
    "fixed_bwar": "Fixed BWAR",
    "local_bwar": "Local BWAR",
}


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(mean: float, se: float, best: float) -> str:
    value = rf"{mean:.3f} ({se:.3f})"
    return rf"\textbf{{{value}}}" if np.isclose(mean, best) else value


def build_s1_table(results: Path, output: Path) -> None:
    summary = pd.read_csv(
        results / "s1_geometry" / "strong_synthetic_transport_summary.csv"
    )
    rows = (
        ("Bures", "Baseline", "Baseline"),
        ("Log-Euclidean", "Baseline", "Log-Euclidean mechanism"),
        ("Cholesky", "Baseline", "Cholesky mechanism"),
        ("Bures", "Shorter series", "Shorter series"),
        ("Bures", "Higher dimension", "Higher dimension"),
        ("Bures", "Weaker dynamics", "Weaker dynamics"),
        ("Bures", "Larger variation", "Larger variation"),
    )
    methods = (
        ("euclidean_gaussian_ar", "Euclidean"),
        ("cholesky_gaussian_ar", "Cholesky"),
        ("log_euclidean_gaussian_ar", "Log-Euclidean"),
        ("bwar_barycenter", "BWAR"),
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Fixed-reference geometry study. Entries are the mean (Monte Carlo standard error) of replication-specific ratios between a method's mean test $W_2^2$ loss and the persistence loss over 80 replications. Persistence is normalized to one and omitted. The first three rows share $T=320$, $d=8$, $\phi=0.70$, and covariance-coordinate dispersion $0.15$ and differ only in the generating covariance chart. The remaining Bures rows perturb the baseline, with the higher-dimensional case jointly changing dimension and dispersion. All methods use one-step forecasts, the same 45\%--20\%--35\% chronological split, ridge grid, and diagonal ridge VAR(1) fit; lower values are better.}",
        r"\label{tab:geometry-robustness}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Generating chart & Setting & $T$ & $d$ & $\phi$ & Disp. & Euclidean AR & Cholesky AR & Log-Euclidean AR & BWAR \\",
        r"\midrule",
    ]
    for chart, setting, design in rows:
        block = summary.loc[summary["design"].eq(design)].copy()
        if len(block) == 0 or set(block["n_rep"]) != {80}:
            raise ValueError(f"S1 design {design!r} must contain 80 replications")
        values = {
            label: (
                float(block.loc[block["method"].eq(method), "w2_ratio_mean"].iloc[0]),
                float(block.loc[block["method"].eq(method), "w2_ratio_se"].iloc[0]),
            )
            for method, label in methods
        }
        best = min(mean for mean, _ in values.values())
        params = block.iloc[0]
        cells = [_cell(*values[label], best) for _, label in methods]
        lines.append(
            f"{chart} & {setting} & {int(params['n'])} & {int(params['d'])} & "
            f"{params['phi']:.2f} & {params['dispersion']:.2f} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    _write(output, lines)


def _origin_bootstrap_se(panel: pd.DataFrame, row: pd.Series) -> float:
    subset = panel.loc[
        panel["method"].eq(row["method"])
        & panel["horizon"].eq(int(row["horizon"]))
    ]
    by_origin = [
        group.sort_values("target_index")[row["metric"]].to_numpy(float)
        for _, group in subset.groupby("origin", sort=True)
    ]
    rng = np.random.default_rng(int(row["seed"]))
    replicates = int(row["replicates"])
    draws = np.zeros(replicates, dtype=float)
    total = sum(len(values) for values in by_origin)
    for values in by_origin:
        block = min(int(row["block_length"]), len(values))
        n_blocks = int(np.ceil(len(values) / block))
        starts = rng.integers(
            0, len(values) - block + 1, size=(replicates, n_blocks)
        )
        indices = starts[:, :, None] + np.arange(block)[None, None, :]
        sampled = values[indices].reshape(replicates, -1)[:, : len(values)]
        draws += sampled.sum(axis=1)
    return float(np.std(draws / total, ddof=1))


def build_divvy_table(results: Path, output: Path) -> None:
    directory = results / "divvy"
    summary = pd.read_csv(directory / "method_level_bootstrap_summary.csv")
    panel = pd.read_csv(directory / "target_level_losses.csv")
    if not set(summary["method"]).issubset(DIVVY_METHODS):
        raise ValueError("Divvy summary contains a method outside the manuscript panel")
    summary = summary.loc[summary["method"].isin(DIVVY_METHODS)].copy()
    summary["bootstrap_se"] = [
        _origin_bootstrap_se(panel, row) for _, row in summary.iterrows()
    ]
    horizons = (3, 4, 5)
    best = {
        (metric, horizon): float(
            summary.loc[
                summary["metric"].eq(metric)
                & summary["horizon"].eq(horizon),
                "mean",
            ].min()
        )
        for metric in ("w2", "raw_rmse")
        for horizon in horizons
    }

    def formatted(method: str, metric: str, horizon: int) -> str:
        row = summary.loc[
            summary["method"].eq(method)
            & summary["metric"].eq(metric)
            & summary["horizon"].eq(horizon)
        ].iloc[0]
        value = rf"{row['mean']:.3f} $\pm$ {row['bootstrap_se']:.3f}"
        return (
            rf"\textbf{{{value}}}"
            if np.isclose(row["mean"], best[(metric, horizon)])
            else value
        )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\caption{Divvy rolling-origin results on 183 common targets per horizon. Entries are target-weighted means $\pm$ origin-preserving moving-block bootstrap standard errors (10,000 resamples). Lower values are better; boldface identifies the smallest point estimate in each endpoint--horizon column.}",
        r"\label{tab:divvy-full-results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Gaussian $W_2^2$ loss} & \multicolumn{3}{c}{Standardized mean RMSE} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & $h=3$ & $h=4$ & $h=5$ & $h=3$ & $h=4$ & $h=5$ \\",
        r"\midrule",
    ]
    for method in DIVVY_METHODS:
        cells = [
            formatted(method, metric, horizon)
            for metric in ("w2", "raw_rmse")
            for horizon in horizons
        ]
        lines.append(DIVVY_LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    _write(output, lines)


def build_pems_table(results: Path, output: Path) -> None:
    directory = results / "pems_bay"
    summary = pd.read_csv(directory / "test_method_summary.csv")
    display = pd.read_csv(directory / "table_display.csv")
    methods = ("Persistence", "Euclidean", "Cholesky", "Log-Euclidean", "BWAR")
    labels = {
        "Persistence": "Persistence",
        "Euclidean": "Euclidean AR",
        "Cholesky": "Cholesky AR",
        "Log-Euclidean": "log-Euclidean AR",
        "BWAR": "Fixed BWAR",
    }
    horizons = (1, 3, 6)
    metrics = (("w2_mean", "w2_se"), ("covariance_mean", "covariance_se"))
    expected_rows = len(methods) * len(horizons)
    if len(display) != expected_rows:
        raise ValueError("PEMS display summary has an incomplete method--horizon panel")
    merged = display.merge(
        summary,
        on=["method", "horizon"],
        how="left",
        validate="one_to_one",
    )
    if merged["w2_squared_mean"].isna().any():
        raise ValueError("PEMS display summary does not match the result panel")
    checks = (
        ("w2_mean", "w2_squared_mean"),
        ("w2_se", "w2_squared_standard_error"),
        ("covariance_mean", "covariance_w2_component_mean"),
        ("covariance_se", "covariance_w2_component_standard_error"),
    )
    for displayed, computed in checks:
        if not np.allclose(merged[displayed], merged[computed], atol=1.1e-3, rtol=0):
            raise ValueError(f"PEMS frozen display column {displayed} changed")
    best = {
        (metric, horizon): float(
            display.loc[display["horizon"].eq(horizon), metric].min()
        )
        for metric, _ in metrics
        for horizon in horizons
    }

    def formatted(method: str, metric: str, se_metric: str, horizon: int) -> str:
        row = display.loc[
            display["method"].eq(method) & display["horizon"].eq(horizon)
        ].iloc[0]
        value = rf"{row[metric]:.3f} $\pm$ {row[se_metric]:.3f}"
        return (
            rf"\textbf{{{value}}}"
            if np.isclose(row[metric], best[(metric, horizon)])
            else value
        )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\caption{PEMS-BAY rolling-origin results averaged over four prespecified geographic panels. Entries are target-weighted means $\pm$ moving-block bootstrap standard errors; lower values are better. Boldface identifies the smallest point estimate in each column. The covariance component identifies the source of the total distributional error.}",
        r"\label{tab:pems-full-results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Gaussian $W_2^2$ loss} & \multicolumn{3}{c}{Covariance Bures component} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & $h=1$ & $h=3$ & $h=6$ & $h=1$ & $h=3$ & $h=6$ \\",
        r"\midrule",
    ]
    for method in methods:
        cells = [
            formatted(method, metric, se_metric, horizon)
            for metric, se_metric in metrics
            for horizon in horizons
        ]
        lines.append(labels[method] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    _write(output, lines)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results" / "reference"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts" / "generated"
    )
    args = parser.parse_args()
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    build_s1_table(
        args.results_root, tables / "geometry_robustness_rebuild.tex"
    )
    build_divvy_table(
        args.results_root, tables / "divvy_full_results_rebuild.tex"
    )
    build_pems_table(args.results_root, tables / "pems_full_results.tex")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_continuous_covariance_regimes.py"),
            "--result-dir",
            str(args.results_root / "s2_continuous_covariance"),
            "--output-stem",
            str(figures / "s2_matched_start_loss"),
        ],
        check=True,
    )
    outputs = sorted(tables.glob("*.tex")) + sorted(figures.glob("*"))
    manifest = {str(path.relative_to(args.output_root)): sha256(path) for path in outputs}
    (args.output_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
