"""Tables, summaries, and figures for the Divvy application."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from bwar.gaussian_geometry import bw_barycenter, project_spd


METHODS = [
    ("Persistence", "persistence", "Pers.", "#A8AFB9"),
    ("Raw VAR", "raw_var_window_ar", "Raw VAR", "#5F6875"),
    ("Euclidean AR", "euclidean_gaussian_ar", "Euc.", "#6F98BF"),
    ("Cholesky AR", "cholesky_gaussian_ar", "Chol.", "#B98D63"),
    ("Log-Euclidean AR", "log_euclidean_gaussian_ar", "LogEuc.", "#927DB8"),
    ("BWAR-barycenter", "bwar_barycenter", "BWAR", "#1F5A9D"),
]
METHOD_ORDER = [method for _, method, _, _ in METHODS]
METHOD_LABEL = {method: label for _, method, label, _ in METHODS}
METHOD_COLOR = {method: color for _, method, _, color in METHODS}


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt_mean_se(mean: float, se: float, best: float | None = None, *, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    out = f"{mean:.{digits}f} ({se:.{digits}f})" if np.isfinite(se) else f"{mean:.{digits}f}"
    if best is not None and np.isclose(float(mean), float(best), rtol=1e-10, atol=1e-12):
        return rf"\textbf{{{out}}}"
    return out


def barycenter_only_references(means_fit: np.ndarray, covs_fit: np.ndarray, *, max_sample_refs: int = 4):
    del max_sample_refs
    covs = np.asarray([project_spd(C, eps=1e-8) for C in covs_fit])
    mean_ref = np.asarray(means_fit, dtype=float).mean(axis=0)
    return [("bw_barycenter", mean_ref, bw_barycenter(covs[: min(len(covs), 250)]))]


def combine_bwar_rows(raw: pd.DataFrame, refs: pd.DataFrame, *, candidate: str) -> pd.DataFrame:
    if candidate != "divvy":
        raise ValueError("the public package contains only the Divvy application")
    parts: list[pd.DataFrame] = []
    if not raw.empty:
        keep = raw.loc[~raw["method"].str.startswith("bwar")].copy()
        keep["candidate"] = candidate
        parts.append(keep)
        selected_reference = raw.get("selected_reference", pd.Series("", index=raw.index)).fillna("")
        bwar = raw.loc[
            raw["method"].eq("bwar_selected_ref") & selected_reference.eq("bw_barycenter")
        ].copy()
        if not bwar.empty:
            bwar["method"] = "bwar_barycenter"
            bwar["candidate"] = candidate
            parts.append(bwar)
    else:
        bwar = pd.DataFrame()
    if bwar.empty and not refs.empty:
        bwar = refs.loc[refs["reference"].eq("bw_barycenter")].copy()
        if not bwar.empty:
            bwar["method"] = "bwar_barycenter"
            bwar["candidate"] = candidate
            bwar["selected_reference"] = "bw_barycenter"
            parts.append(bwar)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["display_dataset"] = "Divvy"
    out["dataset_label"] = "bike-share station trip-count level"
    return out


def available_methods(sub: pd.DataFrame) -> list[str]:
    present = [m for m in METHOD_ORDER if m in set(sub["method"])]
    return [m for m in present if sub.loc[sub["method"].eq(m), "test_domain_loss_mean"].notna().any()]


def summarize_long(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if long.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    group_cols = [
        "candidate",
        "display_dataset",
        "dataset_label",
        "job",
        "window",
        "step",
        "dimension",
        "dimension_arg",
        "q_dimension",
        "h0",
        "horizon",
    ]
    for key, sub in long.groupby(group_cols, dropna=False, sort=False):
        key_data = dict(zip(group_cols, key, strict=False))
        method_stats: dict[str, dict[str, float]] = {}
        for method in available_methods(sub):
            part = sub.loc[sub["method"].eq(method)].copy()
            raw_vals = part["test_domain_loss_mean"].dropna().astype(float)
            w2_vals = part["test_w2_mean"].dropna().astype(float)
            method_stats[method] = {
                "raw_mean": float(raw_vals.mean()) if len(raw_vals) else np.nan,
                "raw_se": float(raw_vals.std(ddof=1) / np.sqrt(len(raw_vals))) if len(raw_vals) > 1 else np.nan,
                "w2_mean": float(w2_vals.mean()) if len(w2_vals) else np.nan,
                "w2_se": float(w2_vals.std(ddof=1) / np.sqrt(len(w2_vals))) if len(w2_vals) > 1 else np.nan,
                "n_origins": int(part["origin"].nunique()),
            }
            method_rows.append({**key_data, "method": method, **method_stats[method]})
        if "bwar_barycenter" not in method_stats:
            continue
        raw_valid = {m: v["raw_mean"] for m, v in method_stats.items() if np.isfinite(v["raw_mean"])}
        w2_valid = {m: v["w2_mean"] for m, v in method_stats.items() if np.isfinite(v["w2_mean"])}
        if "bwar_barycenter" not in raw_valid:
            continue
        non_bwar_raw = {m: v for m, v in raw_valid.items() if m != "bwar_barycenter"}
        non_bwar_w2 = {m: v for m, v in w2_valid.items() if m != "bwar_barycenter"}
        if not non_bwar_raw:
            continue
        best_raw_method, best_raw = min(raw_valid.items(), key=lambda kv: kv[1])
        best_w2_method, best_w2 = min(w2_valid.items(), key=lambda kv: kv[1]) if w2_valid else ("", np.nan)
        best_non_raw_method, best_non_raw = min(non_bwar_raw.items(), key=lambda kv: kv[1])
        best_non_w2_method, best_non_w2 = (
            min(non_bwar_w2.items(), key=lambda kv: kv[1]) if non_bwar_w2 else ("", np.nan)
        )
        bwar_raw = raw_valid["bwar_barycenter"]
        bwar_w2 = w2_valid.get("bwar_barycenter", np.nan)
        rows.append(
            {
                **key_data,
                "bwar_raw_mean": float(bwar_raw),
                "bwar_raw_se": float(method_stats["bwar_barycenter"]["raw_se"]),
                "bwar_w2_mean": float(bwar_w2),
                "bwar_w2_se": float(method_stats["bwar_barycenter"]["w2_se"]),
                "best_raw_method": best_raw_method,
                "best_raw_mean": float(best_raw),
                "best_w2_method": best_w2_method,
                "best_w2_mean": float(best_w2) if np.isfinite(best_w2) else np.nan,
                "best_non_bwar_raw_method": best_non_raw_method,
                "best_non_bwar_raw_mean": float(best_non_raw),
                "best_non_bwar_w2_method": best_non_w2_method,
                "best_non_bwar_w2_mean": float(best_non_w2) if np.isfinite(best_non_w2) else np.nan,
                "bwar_best_raw": best_raw_method == "bwar_barycenter",
                "bwar_best_w2": best_w2_method == "bwar_barycenter",
                "raw_margin": float(best_non_raw - bwar_raw),
                "raw_margin_fraction": float((best_non_raw - bwar_raw) / max(abs(best_non_raw), 1e-12)),
                "w2_margin": float(best_non_w2 - bwar_w2) if np.isfinite(best_non_w2) else np.nan,
                "w2_margin_fraction": float((best_non_w2 - bwar_w2) / max(abs(best_non_w2), 1e-12))
                if np.isfinite(best_non_w2) and np.isfinite(bwar_w2)
                else np.nan,
                "n_origins": int(method_stats["bwar_barycenter"]["n_origins"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(method_rows)


def write_application_table(method_summary: pd.DataFrame, case: pd.Series, path: Path) -> None:
    key = (
        method_summary["candidate"].eq(str(case["candidate"]))
        & method_summary["window"].eq(int(case["window"]))
        & method_summary["step"].eq(int(case["step"]))
        & method_summary["dimension"].eq(int(case["dimension"]))
        & method_summary["horizon"].eq(int(case["horizon"]))
    )
    part = method_summary.loc[key].copy()
    if part.empty:
        raise ValueError("application table has no method rows")
    methods = [m for m in METHOD_ORDER if m in set(part["method"])]
    raw_best = float(part["raw_mean"].min())
    w2_best = float(part["w2_mean"].min())
    setting = (
        rf"\(w={int(case['window'])},\ \mathrm{{step}}={int(case['step'])},\ "
        rf"d={int(case['dimension'])},\ q={int(case['q_dimension'])},\ h={int(case['horizon'])}\)"
    )
    rows = []
    for endpoint, mean_col, se_col, best in [
        ("Raw window-mean RMSE", "raw_mean", "raw_se", raw_best),
        (r"Gaussian \(W_2^2\) loss", "w2_mean", "w2_se", w2_best),
    ]:
        cells = []
        for method in methods:
            row = part.loc[part["method"].eq(method)].iloc[0]
            cells.append(fmt_mean_se(float(row[mean_col]), float(row[se_col]), best=best))
        rows.append((endpoint, cells))
    align = "ll" + "r" * len(methods)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{tex_escape(str(case['display_dataset']))} real-data application. Entries are mean (standard error) over chronological rolling-origin blocks; lower is better. The primary endpoint is training-standardized physical-mean RMSE, and the second endpoint is the same-task Gaussian \(W_2^2\) loss.}}",
        r"\label{tab:redone-realdata-application}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        "Endpoint & Setting & " + " & ".join(METHOD_LABEL[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for endpoint, cells in rows:
        lines.append(f"{endpoint} & {setting} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_horizon_table(method_summary: pd.DataFrame, case: pd.Series, path: Path) -> None:
    key = (
        method_summary["candidate"].eq(str(case["candidate"]))
        & method_summary["window"].eq(int(case["window"]))
        & method_summary["step"].eq(int(case["step"]))
        & method_summary["dimension"].eq(int(case["dimension"]))
    )
    part = method_summary.loc[key].copy()
    methods = [m for m in METHOD_ORDER if m in set(part["method"]) and m != "persistence"]
    horizons = sorted(int(h) for h in part["horizon"].unique())
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{tex_escape(str(case['display_dataset']))} horizon sensitivity for the standardized physical-mean endpoint. Entries are mean (standard error) over chronological rolling-origin blocks; lower is better.}}",
        r"\label{tab:redone-realdata-horizon}",
        r"\resizebox{0.92\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(methods) + r"}",
        r"\toprule",
        "Horizon & " + " & ".join(METHOD_LABEL[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for horizon in horizons:
        block = part.loc[part["horizon"].eq(horizon)]
        best = float(block["raw_mean"].min())
        cells = []
        for method in methods:
            row = block.loc[block["method"].eq(method)]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                cells.append(fmt_mean_se(float(r["raw_mean"]), float(r["raw_se"]), best=best))
        lines.append(rf"\(h={horizon}\) & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-bwar")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    return plt


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=9.4, fontweight="bold", va="bottom")


def make_application_figure(long: pd.DataFrame, method_summary: pd.DataFrame, case: pd.Series, path_stem: Path) -> None:
    plt = import_matplotlib()
    key = (
        method_summary["candidate"].eq(str(case["candidate"]))
        & method_summary["window"].eq(int(case["window"]))
        & method_summary["step"].eq(int(case["step"]))
        & method_summary["dimension"].eq(int(case["dimension"]))
    )
    summary = method_summary.loc[key].copy()
    methods = [m for m in METHOD_ORDER if m in set(summary["method"]) and m != "persistence"]
    horizons = sorted(int(h) for h in summary["horizon"].unique())

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_w2 = fig.add_subplot(gs[0, 1])
    ax_origin = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    for ax, mean_col, se_col, ylabel, title in [
        (ax_raw, "raw_mean", "raw_se", "Raw window-mean RMSE", "Physical endpoint"),
        (ax_w2, "w2_mean", "w2_se", r"Gaussian $W_2^2$ loss", "Distributional endpoint"),
    ]:
        for method in methods:
            part = summary.loc[summary["method"].eq(method)].sort_values("horizon")
            ax.errorbar(
                part["horizon"].astype(int),
                part[mean_col].astype(float),
                yerr=part[se_col].astype(float),
                marker="o",
                ms=4.0,
                lw=2.05 if method == "bwar_barycenter" else 1.35,
                capsize=2.0,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
                zorder=3 if method == "bwar_barycenter" else 2,
            )
        ax.set_xticks(horizons)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E7EB", lw=0.55)
    ax_w2.set_yscale("log")

    main_h = int(case["horizon"])
    long_key = (
        long["candidate"].eq(str(case["candidate"]))
        & long["window"].eq(int(case["window"]))
        & long["step"].eq(int(case["step"]))
        & long["dimension"].eq(int(case["dimension"]))
        & long["horizon"].eq(main_h)
        & long["method"].isin(methods)
    )
    hlong = long.loc[long_key].copy()
    origins = sorted(int(o) for o in hlong["origin"].unique())
    x = np.arange(len(origins))
    offsets = np.linspace(-0.24, 0.24, max(len(methods), 1))
    for off, method in zip(offsets, methods):
        part = hlong.loc[hlong["method"].eq(method)].sort_values("origin")
        ax_origin.plot(
            x + off,
            part["test_domain_loss_mean"].astype(float),
            "o-",
            lw=2.0 if method == "bwar_barycenter" else 1.2,
            ms=4.0,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
        )
    ax_origin.set_xticks(x)
    ax_origin.set_xticklabels([str(o) for o in origins])
    ax_origin.set_xlabel("Rolling-origin block")
    ax_origin.set_ylabel("Raw window-mean RMSE")
    ax_origin.set_title(rf"Chronological blocks at $h={main_h}$", loc="left", fontweight="bold")
    ax_origin.grid(axis="y", color="#E5E7EB", lw=0.55)

    hsummary = summary.loc[summary["horizon"].eq(main_h) & summary["method"].isin(methods)].copy()
    hsummary["order"] = hsummary["method"].map({m: i for i, m in enumerate(methods)})
    hsummary = hsummary.sort_values("order")
    ax_bar.bar(
        np.arange(len(hsummary)),
        hsummary["raw_mean"].astype(float),
        yerr=hsummary["raw_se"].astype(float),
        color=[METHOD_COLOR[m] for m in hsummary["method"]],
        edgecolor="#263445",
        linewidth=0.55,
        capsize=2.2,
        width=0.62,
    )
    ax_bar.set_xticks(np.arange(len(hsummary)))
    ax_bar.set_xticklabels([METHOD_LABEL[m] for m in hsummary["method"]], rotation=25, ha="right")
    ax_bar.set_ylabel("Raw window-mean RMSE")
    ax_bar.set_title(rf"Mean over origins at $h={main_h}$", loc="left", fontweight="bold")
    ax_bar.grid(axis="y", color="#E5E7EB", lw=0.55)

    for label, ax in zip("abcd", [ax_raw, ax_w2, ax_origin, ax_bar], strict=False):
        panel_label(ax, label)
    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.010), ncol=min(5, len(methods)))
    fig.subplots_adjust(bottom=0.16, top=0.94)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=420, bbox_inches="tight")
    plt.close(fig)
