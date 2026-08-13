from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.gaussian_geometry import (  # noqa: E402
    bw2_cov,
    bw_barycenter,
    mat_exp,
    mat_from_triu,
    mat_log,
    ot_map,
    project_spd,
    triu_vec,
)
from bwar.paper_jcgs.gaussian_models import fit_var  # noqa: E402


OUT_DIR = ROOT / "results" / "generated" / "s1_geometry"
OVERLEAF = ROOT / "artifacts" / "generated"
TABLE_DIR = OVERLEAF / "tables"
FIGURE_DIR = OVERLEAF / "figures"

DEFAULT_RIDGE_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
METHODS = [
    ("Persistence", "persistence", "Pers.", "#A8AFB9"),
    ("Euclidean AR", "euclidean_gaussian_ar", "Euc.", "#6F98BF"),
    ("Cholesky AR", "cholesky_gaussian_ar", "Chol.", "#B98D63"),
    ("Log-Euclidean AR", "log_euclidean_gaussian_ar", "LogEuc.", "#927DB8"),
    ("BWAR", "bwar_barycenter", "BWAR", "#1F5A9D"),
]
PLOT_METHODS = ["euclidean_gaussian_ar", "cholesky_gaussian_ar", "log_euclidean_gaussian_ar", "bwar_barycenter"]
METHOD_LABEL = {code: short for _, code, short, _ in METHODS}
METHOD_COLOR = {code: color for _, code, _, color in METHODS}


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt_mean_se(mean: float, se: float, *, best: float | None = None, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    out = f"{mean:.{digits}f} ({se:.{digits}f})" if np.isfinite(se) else f"{mean:.{digits}f}"
    if best is not None and np.isclose(float(mean), float(best), rtol=1e-10, atol=1e-12):
        return rf"\textbf{{{out}}}"
    return out


def split_indices(n: int) -> tuple[int, int, int]:
    fit_end = max(40, int(0.45 * n))
    val_end = max(fit_end + 20, int(0.65 * n))
    val_end = min(val_end, n - 20)
    return fit_end, val_end, n


def random_reference(rng: np.random.Generator, d: int, *, condition: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.exp(np.linspace(-0.5 * np.log(condition), 0.5 * np.log(condition), d))
    rng.shuffle(eigs)
    cov = project_spd((q * eigs) @ q.T, eps=1e-8)
    mean = rng.normal(scale=0.25, size=d)
    return mean, cov


def safe_transport_from_vech(
    z: np.ndarray,
    d: int,
    *,
    min_transport_eig: float = 0.18,
) -> np.ndarray:
    H = mat_from_triu(z, d)
    vals = np.linalg.eigvalsh(np.eye(d) + H)
    if vals.min() < min_transport_eig:
        h_min = float(np.linalg.eigvalsh(H).min())
        if h_min < 0:
            scale = (1.0 - min_transport_eig) / max(abs(h_min), 1e-12)
            H = scale * H
    return project_spd(np.eye(d) + H, eps=1e-8)


def project_to_spectral_shell(C: np.ndarray, *, lo: float = 0.2, hi: float = 5.0) -> np.ndarray:
    vals, vecs = np.linalg.eigh(project_spd(C, eps=1e-8))
    vals = np.clip(vals, lo, hi)
    return project_spd((vecs * vals) @ vecs.T, eps=1e-8)


def active_coordinate_sets(d: int, rng: np.random.Generator, *, active_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_i, tri_j = np.triu_indices(d)
    diag = np.flatnonzero(tri_i == tri_j)
    offdiag = np.flatnonzero(tri_i != tri_j)
    n_mean = max(2, int(np.ceil(active_fraction * d)))
    n_diag = max(2, int(np.ceil(0.35 * len(diag))))
    n_offdiag = max(4, int(np.ceil(active_fraction * len(offdiag))))
    active_mean = rng.choice(np.arange(d), size=min(n_mean, d), replace=False)
    active_diag = rng.choice(diag, size=min(n_diag, len(diag)), replace=False)
    active_offdiag = rng.choice(offdiag, size=min(n_offdiag, len(offdiag)), replace=False)
    return active_mean, active_diag, active_offdiag


def simulate_transport_linear_gaussians(
    *,
    n: int,
    d: int,
    phi: float,
    dispersion: float,
    seed: int,
    mean_dispersion: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(9157 + seed)
    ref_mean, ref_cov = random_reference(rng, d, condition=8.0)
    q_cov = d * (d + 1) // 2
    q_total = d + q_cov
    active_mean, active_diag, active_offdiag = active_coordinate_sets(d, rng, active_fraction=0.35)
    active_cov = np.r_[active_diag, active_offdiag]
    active_total = np.r_[active_mean, d + active_cov]

    slopes = np.zeros(q_total, dtype=float)
    slopes[active_total] = np.clip(phi * rng.uniform(0.985, 1.015, size=len(active_total)), 0.05, 0.985)
    stationary_sd = np.full(q_total, 0.002, dtype=float)
    stationary_sd[active_mean] = mean_dispersion
    stationary_sd[d + active_cov] = dispersion
    innovation = stationary_sd * np.sqrt(np.maximum(1.0 - slopes**2, 0.05))

    z = np.zeros((n, q_total), dtype=float)
    z[0] = rng.normal(scale=stationary_sd)
    for t in range(1, n):
        z[t] = slopes * z[t - 1] + rng.normal(scale=innovation)

    means = np.empty((n, d), dtype=float)
    covs = np.empty((n, d, d), dtype=float)
    transport_clip_count = 0
    shell_clip_count = 0
    for t in range(n):
        u_t = z[t, :d]
        v_t = z[t, d:]
        raw_transport = np.eye(d) + mat_from_triu(v_t, d)
        transport_clip_count += int(
            np.linalg.eigvalsh(raw_transport).min() < 0.05
        )
        A_t = safe_transport_from_vech(v_t, d, min_transport_eig=0.05)
        means[t] = ref_mean + u_t
        raw_covariance = A_t @ ref_cov @ A_t.T
        raw_eigenvalues = np.linalg.eigvalsh(raw_covariance)
        shell_clip_count += int(
            raw_eigenvalues.min() < 0.05 or raw_eigenvalues.max() > 20.0
        )
        covs[t] = project_to_spectral_shell(raw_covariance, lo=0.05, hi=20.0)

    meta = {
        "generating_coordinate": "bures_transport",
        "q_total": int(q_total),
        "q_cov": int(q_cov),
        "n_active_mean": int(len(active_mean)),
        "n_active_diag": int(len(active_diag)),
        "n_active_offdiag": int(len(active_offdiag)),
        "reference_condition": 8.0,
        "mean_dispersion": float(mean_dispersion),
        "spectral_shell_low": 0.05,
        "spectral_shell_high": 20.0,
        "generator_transport_clip_rate": float(transport_clip_count / n),
        "generator_shell_clip_rate": float(shell_clip_count / n),
    }
    return means, covs, ref_mean, ref_cov, meta


def simulate_log_euclidean_linear_gaussians(
    *,
    n: int,
    d: int,
    phi: float,
    dispersion: float,
    seed: int,
    mean_dispersion: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Generate sparse AR dynamics in a Log-Euclidean covariance chart."""

    rng = np.random.default_rng(9157 + seed)
    ref_mean, ref_cov = random_reference(rng, d, condition=8.0)
    q_cov = d * (d + 1) // 2
    q_total = d + q_cov
    active_mean, active_diag, active_offdiag = active_coordinate_sets(
        d,
        rng,
        active_fraction=0.35,
    )
    active_cov = np.r_[active_diag, active_offdiag]
    active_total = np.r_[active_mean, d + active_cov]

    slopes = np.zeros(q_total, dtype=float)
    slopes[active_total] = np.clip(
        phi * rng.uniform(0.985, 1.015, size=len(active_total)),
        0.05,
        0.985,
    )
    stationary_sd = np.full(q_total, 0.002, dtype=float)
    stationary_sd[active_mean] = mean_dispersion
    stationary_sd[d + active_cov] = dispersion
    innovation = stationary_sd * np.sqrt(
        np.maximum(1.0 - slopes**2, 0.05)
    )

    coordinates = np.zeros((n, q_total), dtype=float)
    coordinates[0] = rng.normal(scale=stationary_sd)
    for index in range(1, n):
        coordinates[index] = (
            slopes * coordinates[index - 1]
            + rng.normal(scale=innovation)
        )

    reference_log_covariance = mat_log(ref_cov)
    means = np.empty((n, d), dtype=float)
    covariances = np.empty((n, d, d), dtype=float)
    for index in range(n):
        means[index] = ref_mean + coordinates[index, :d]
        log_increment = mat_from_triu(coordinates[index, d:], d)
        covariances[index] = project_to_spectral_shell(
            mat_exp(reference_log_covariance + log_increment),
            lo=0.05,
            hi=20.0,
        )

    metadata = {
        "generating_coordinate": "log_euclidean",
        "q_total": int(q_total),
        "q_cov": int(q_cov),
        "n_active_mean": int(len(active_mean)),
        "n_active_diag": int(len(active_diag)),
        "n_active_offdiag": int(len(active_offdiag)),
        "reference_condition": 8.0,
        "mean_dispersion": float(mean_dispersion),
        "spectral_shell_low": 0.05,
        "spectral_shell_high": 20.0,
    }
    return means, covariances, ref_mean, ref_cov, metadata


def simulate_cholesky_linear_gaussians(
    *,
    n: int,
    d: int,
    phi: float,
    dispersion: float,
    seed: int,
    mean_dispersion: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Generate sparse AR dynamics in log-diagonal Cholesky coordinates."""

    rng = np.random.default_rng(9157 + seed)
    ref_mean, ref_cov = random_reference(rng, d, condition=8.0)
    lower_rows, lower_cols = np.tril_indices(d)
    diagonal_positions = np.flatnonzero(lower_rows == lower_cols)
    off_diagonal_positions = np.flatnonzero(lower_rows != lower_cols)
    q_cov = len(lower_rows)
    q_total = d + q_cov

    n_active_mean = max(2, int(np.ceil(0.35 * d)))
    n_active_diagonal = max(
        2,
        int(np.ceil(0.35 * len(diagonal_positions))),
    )
    n_active_off_diagonal = max(
        4,
        int(np.ceil(0.35 * len(off_diagonal_positions))),
    )
    active_mean = rng.choice(
        np.arange(d),
        size=min(n_active_mean, d),
        replace=False,
    )
    active_diagonal = rng.choice(
        diagonal_positions,
        size=min(n_active_diagonal, len(diagonal_positions)),
        replace=False,
    )
    active_off_diagonal = rng.choice(
        off_diagonal_positions,
        size=min(n_active_off_diagonal, len(off_diagonal_positions)),
        replace=False,
    )
    active_covariance = np.r_[active_diagonal, active_off_diagonal]
    active_total = np.r_[active_mean, d + active_covariance]

    slopes = np.zeros(q_total, dtype=float)
    slopes[active_total] = np.clip(
        phi * rng.uniform(0.985, 1.015, size=len(active_total)),
        0.05,
        0.985,
    )
    stationary_sd = np.full(q_total, 0.002, dtype=float)
    stationary_sd[active_mean] = mean_dispersion
    stationary_sd[d + active_covariance] = dispersion
    innovation_sd = stationary_sd * np.sqrt(
        np.maximum(1.0 - slopes**2, 0.05)
    )

    coordinates = np.zeros((n, q_total), dtype=float)
    coordinates[0] = rng.normal(scale=stationary_sd)
    for index in range(1, n):
        coordinates[index] = (
            slopes * coordinates[index - 1]
            + rng.normal(scale=innovation_sd)
        )

    reference_cholesky_coordinate = cholesky_encode(
        ref_mean,
        ref_cov,
    )[d:]
    means = np.empty((n, d), dtype=float)
    covariances = np.empty((n, d, d), dtype=float)
    shell_clip_count = 0
    for index in range(n):
        means[index] = ref_mean + coordinates[index, :d]
        _, raw_covariance = cholesky_decode(
            np.r_[
                means[index],
                reference_cholesky_coordinate + coordinates[index, d:],
            ],
            d,
        )
        eigenvalues = np.linalg.eigvalsh(raw_covariance)
        if float(eigenvalues.min()) < 0.05 or float(eigenvalues.max()) > 20.0:
            shell_clip_count += 1
        covariances[index] = project_to_spectral_shell(
            raw_covariance,
            lo=0.05,
            hi=20.0,
        )

    metadata = {
        "generating_coordinate": "cholesky",
        "q_total": int(q_total),
        "q_cov": int(q_cov),
        "n_active_mean": int(len(active_mean)),
        "n_active_diag": int(len(active_diagonal)),
        "n_active_offdiag": int(len(active_off_diagonal)),
        "reference_condition": 8.0,
        "mean_dispersion": float(mean_dispersion),
        "spectral_shell_low": 0.05,
        "spectral_shell_high": 20.0,
        "generator_shell_clip_rate": float(shell_clip_count / n),
    }
    return means, covariances, ref_mean, ref_cov, metadata


def gaussian_w2_squared(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> tuple[float, float, float]:
    mean_loss = float(np.sum((np.asarray(mean_a) - np.asarray(mean_b)) ** 2))
    cov_loss = float(bw2_cov(cov_a, cov_b))
    return mean_loss + cov_loss, mean_loss, cov_loss


def recursive_predict(z0: np.ndarray, W: np.ndarray, horizon: int = 1) -> np.ndarray:
    z = np.asarray(z0, dtype=float)
    for _ in range(horizon):
        z = np.r_[1.0, z] @ W
    return z


def score_encoded_forecasts(
    means: np.ndarray,
    covs: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    decode,
    *,
    start_t: int,
    stop_t: int,
    horizon: int = 1,
) -> dict[str, float | int]:
    total_losses: list[float] = []
    mean_losses: list[float] = []
    cov_losses: list[float] = []
    min_eigs: list[float] = []
    for t in range(start_t, stop_t):
        if t + horizon >= len(covs):
            continue
        pred_mean, pred_cov = decode(recursive_predict(Z[t], W, horizon=horizon))
        pred_cov = project_spd(pred_cov, eps=1e-8)
        total, mean_part, cov_part = gaussian_w2_squared(pred_mean, pred_cov, means[t + horizon], covs[t + horizon])
        total_losses.append(total)
        mean_losses.append(mean_part)
        cov_losses.append(cov_part)
        min_eigs.append(float(np.linalg.eigvalsh(pred_cov).min()))
    total_arr = np.asarray(total_losses, dtype=float)
    mean_arr = np.asarray(mean_losses, dtype=float)
    cov_arr = np.asarray(cov_losses, dtype=float)
    return {
        "w2_mean": float(total_arr.mean()),
        "w2_median": float(np.median(total_arr)),
        "w2_q90": float(np.quantile(total_arr, 0.9)),
        "mean_component": float(mean_arr.mean()),
        "cov_component": float(cov_arr.mean()),
        "n_pairs": int(len(total_arr)),
        "min_pred_eig": float(np.min(min_eigs)) if min_eigs else np.nan,
    }


def select_encoded_ar(
    means: np.ndarray,
    covs: np.ndarray,
    encode,
    decode,
    *,
    fit_end: int,
    val_end: int,
    ar_model: str = "diag",
    horizon: int = 1,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
) -> dict[str, float | int]:
    Z = np.vstack([encode(m, C) for m, C in zip(means, covs)])
    val_start = max(0, fit_end - horizon)
    val_stop = max(val_start, val_end - horizon)
    best_lam = ridge_grid[0]
    best_score = np.inf
    best_val: dict[str, float | int] | None = None
    for lam in ridge_grid:
        W = fit_var(Z, fit_end, lam=lam, model=ar_model)
        val_metrics = score_encoded_forecasts(
            means,
            covs,
            Z,
            W,
            decode,
            start_t=val_start,
            stop_t=val_stop,
            horizon=horizon,
        )
        if float(val_metrics["w2_mean"]) < best_score:
            best_score = float(val_metrics["w2_mean"])
            best_lam = lam
            best_val = val_metrics
    if best_val is None:
        raise RuntimeError("no finite validation score")
    final_W = fit_var(Z, val_end, lam=best_lam, model=ar_model)
    test_metrics = score_encoded_forecasts(
        means,
        covs,
        Z,
        final_W,
        decode,
        start_t=max(0, val_end - horizon),
        stop_t=len(covs) - horizon,
        horizon=horizon,
    )
    return {
        "ridge": float(best_lam),
        "val_w2_mean": float(best_val["w2_mean"]),
        "test_w2_mean": float(test_metrics["w2_mean"]),
        "test_w2_median": float(test_metrics["w2_median"]),
        "test_w2_q90": float(test_metrics["w2_q90"]),
        "test_mean_component": float(test_metrics["mean_component"]),
        "test_cov_component": float(test_metrics["cov_component"]),
        "n_test_pairs": int(test_metrics["n_pairs"]),
        "min_pred_eig": float(test_metrics["min_pred_eig"]),
    }


def persistence_metrics(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int = 1,
) -> dict[str, float | int]:
    def score(start_t: int, stop_t: int) -> dict[str, float | int]:
        total_losses = []
        mean_losses = []
        cov_losses = []
        for t in range(start_t, stop_t):
            if t + horizon >= len(covs):
                continue
            total, mean_part, cov_part = gaussian_w2_squared(means[t], covs[t], means[t + horizon], covs[t + horizon])
            total_losses.append(total)
            mean_losses.append(mean_part)
            cov_losses.append(cov_part)
        total_arr = np.asarray(total_losses, dtype=float)
        mean_arr = np.asarray(mean_losses, dtype=float)
        cov_arr = np.asarray(cov_losses, dtype=float)
        return {
            "w2_mean": float(total_arr.mean()),
            "mean_component": float(mean_arr.mean()),
            "cov_component": float(cov_arr.mean()),
            "n_pairs": int(len(total_arr)),
        }

    val = score(max(0, fit_end - horizon), max(0, val_end - horizon))
    test = score(max(0, val_end - horizon), len(covs) - horizon)
    return {
        "ridge": np.nan,
        "val_w2_mean": float(val["w2_mean"]),
        "test_w2_mean": float(test["w2_mean"]),
        "test_w2_median": np.nan,
        "test_w2_q90": np.nan,
        "test_mean_component": float(test["mean_component"]),
        "test_cov_component": float(test["cov_component"]),
        "n_test_pairs": int(test["n_pairs"]),
        "min_pred_eig": float(min(np.linalg.eigvalsh(C).min() for C in covs[max(0, val_end - horizon) : -horizon])),
    }


def cholesky_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = project_spd(cov, eps=1e-8)
    L = np.linalg.cholesky(cov)
    idx = np.tril_indices(cov.shape[0])
    z = L[idx].copy()
    diag_pos = [np.where((idx[0] == j) & (idx[1] == j))[0][0] for j in range(cov.shape[0])]
    z[diag_pos] = np.log(np.clip(z[diag_pos], 1e-12, None))
    return np.r_[mean, z]


def cholesky_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(z[:d], dtype=float)
    lower = np.asarray(z[d:], dtype=float)
    L = np.zeros((d, d), dtype=float)
    idx = np.tril_indices(d)
    L[idx] = lower
    for j in range(d):
        L[j, j] = np.exp(np.clip(L[j, j], -30, 30))
    return mean, project_spd(L @ L.T, eps=1e-8)


def bwar_encode(mean: np.ndarray, cov: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray) -> np.ndarray:
    A = ot_map(ref_cov, cov)
    return np.r_[np.asarray(mean) - np.asarray(ref_mean), triu_vec(A - np.eye(ref_cov.shape[0]))]


def bwar_decode(z: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = ref_cov.shape[0]
    mean = np.asarray(ref_mean) + np.asarray(z[:d])
    A = safe_transport_from_vech(np.asarray(z[d:]), d, min_transport_eig=0.08)
    return mean, project_spd(A @ ref_cov @ A.T, eps=1e-8)


def run_setting(
    *,
    design: str,
    n: int,
    d: int,
    phi: float,
    dispersion: float,
    seed: int,
    ar_model: str = "diag",
    generating_coordinate: str = "bures_transport",
) -> pd.DataFrame:
    generators = {
        "bures_transport": simulate_transport_linear_gaussians,
        "log_euclidean": simulate_log_euclidean_linear_gaussians,
        "cholesky": simulate_cholesky_linear_gaussians,
    }
    if generating_coordinate not in generators:
        raise ValueError(
            f"unknown generating coordinate: {generating_coordinate}"
        )
    means, covs, ref_mean, ref_cov, meta = generators[
        generating_coordinate
    ](
        n=n,
        d=d,
        phi=phi,
        dispersion=dispersion,
        seed=seed,
    )
    fit_end, val_end, test_end = split_indices(n)
    rows: list[dict[str, object]] = []

    def add(method: str, metrics: dict[str, object]) -> None:
        rows.append(
            {
                "design": design,
                "seed": int(seed),
                "n": int(n),
                "d": int(d),
                "phi": float(phi),
                "dispersion": float(dispersion),
                "fit_end": int(fit_end),
                "val_end": int(val_end),
                "test_end": int(test_end),
                "ar_model": ar_model,
                "method": method,
                **meta,
                **metrics,
            }
        )

    add("persistence", persistence_metrics(means, covs, fit_end=fit_end, val_end=val_end))
    add(
        "euclidean_gaussian_ar",
        select_encoded_ar(
            means,
            covs,
            lambda m, C: np.r_[m, triu_vec(C)],
            lambda z, dim=d: (np.asarray(z[:dim]), project_spd(mat_from_triu(np.asarray(z[dim:]), dim), eps=1e-8)),
            fit_end=fit_end,
            val_end=val_end,
            ar_model=ar_model,
        ),
    )
    add(
        "cholesky_gaussian_ar",
        select_encoded_ar(
            means,
            covs,
            cholesky_encode,
            lambda z, dim=d: cholesky_decode(z, dim),
            fit_end=fit_end,
            val_end=val_end,
            ar_model=ar_model,
        ),
    )
    add(
        "log_euclidean_gaussian_ar",
        select_encoded_ar(
            means,
            covs,
            lambda m, C: np.r_[m, triu_vec(mat_log(C))],
            lambda z, dim=d: (np.asarray(z[:dim]), mat_exp(mat_from_triu(np.asarray(z[dim:]), dim))),
            fit_end=fit_end,
            val_end=val_end,
            ar_model=ar_model,
        ),
    )
    bary_cov = bw_barycenter(covs[:fit_end])
    bary_mean = means[:fit_end].mean(axis=0)
    add(
        "bwar_barycenter",
        select_encoded_ar(
            means,
            covs,
            lambda m, C, rm=bary_mean, rc=bary_cov: bwar_encode(m, C, rm, rc),
            lambda z, rm=bary_mean, rc=bary_cov: bwar_decode(z, rm, rc),
            fit_end=fit_end,
            val_end=val_end,
            ar_model=ar_model,
        ),
    )

    out = pd.DataFrame(rows)
    persistence = out.loc[out["method"].eq("persistence")].iloc[0]
    out["w2_ratio_to_persistence"] = out["test_w2_mean"].astype(float) / max(float(persistence["test_w2_mean"]), 1e-12)
    out["cov_ratio_to_persistence"] = out["test_cov_component"].astype(float) / max(float(persistence["test_cov_component"]), 1e-12)
    out["mean_ratio_to_persistence"] = out["test_mean_component"].astype(float) / max(float(persistence["test_mean_component"]), 1e-12)
    return out


def default_settings() -> list[dict[str, object]]:
    return [
        # The smaller covariance-coordinate scales keep the transport generator
        # inside the SPD shell with only rare admissibility projections.  This
        # preserves the intended transport-linear mechanism instead of making
        # the projection itself a dominant part of the data-generating process.
        {"design": "Baseline", "n": 320, "d": 8, "phi": 0.70, "dispersion": 0.15},
        {"design": "Shorter series", "n": 200, "d": 8, "phi": 0.70, "dispersion": 0.15},
        {"design": "Higher dimension", "n": 320, "d": 10, "phi": 0.70, "dispersion": 0.12},
        {"design": "Weaker dynamics", "n": 320, "d": 8, "phi": 0.50, "dispersion": 0.15},
        {"design": "Larger variation", "n": 320, "d": 8, "phi": 0.70, "dispersion": 0.18},
        {
            "design": "Log-Euclidean mechanism",
            "n": 320,
            "d": 8,
            "phi": 0.70,
            "dispersion": 0.15,
            "generating_coordinate": "log_euclidean",
        },
        {
            "design": "Cholesky mechanism",
            "n": 320,
            "d": 8,
            "phi": 0.70,
            "dispersion": 0.15,
            "generating_coordinate": "cholesky",
        },
    ]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["design", "n", "d", "phi", "dispersion", "method"], as_index=False)
        .agg(
            n_rep=("seed", "nunique"),
            w2_ratio_mean=("w2_ratio_to_persistence", "mean"),
            w2_ratio_se=("w2_ratio_to_persistence", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            cov_ratio_mean=("cov_ratio_to_persistence", "mean"),
            cov_ratio_se=("cov_ratio_to_persistence", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            mean_ratio_mean=("mean_ratio_to_persistence", "mean"),
            mean_ratio_se=("mean_ratio_to_persistence", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            min_pred_eig=("min_pred_eig", "min"),
        )
        .sort_values(["design", "method"])
    )


def write_main_table(
    summary: pd.DataFrame,
    path: Path,
    *,
    reps: int | None = None,
    ar_model: str = "diag",
) -> None:
    part = summary.loc[summary["design"].eq("Baseline")].copy()
    stats = {row["method"]: row for _, row in part.iterrows()}
    method_codes = [code for _, code, _, _ in METHODS]
    best_w2 = min(float(stats[m]["w2_ratio_mean"]) for m in method_codes if m in stats)
    best_cov = min(float(stats[m]["cov_ratio_mean"]) for m in method_codes if m in stats)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Transport-linear Gaussian simulation. Entries are mean test-loss ratios to persistence over {reps if reps is not None else 50} replications, with standard errors in parentheses. Lower values are better. The design uses \(d=8\), \(T=320\), a non-identity Bures reference, and {ar_model} ridge VAR(1) lag-design fits.}}",
        r"\label{tab:synthetic-transport-main}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Full \(W_2^2\) ratio & Covariance ratio \\",
        r"\midrule",
    ]
    for full, method, _, _ in METHODS:
        row = stats[method]
        lines.append(
            f"{full} & "
            f"{fmt_mean_se(float(row['w2_ratio_mean']), float(row['w2_ratio_se']), best=best_w2)} & "
            f"{fmt_mean_se(float(row['cov_ratio_mean']), float(row['cov_ratio_se']), best=best_cov)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_variation_table(
    summary: pd.DataFrame,
    path: Path,
    *,
    reps: int | None = None,
    ar_model: str = "diag",
) -> None:
    designs = [
        "Baseline",
        "Shorter series",
        "Higher dimension",
        "Weaker dynamics",
        "Larger variation",
        "Log-Euclidean mechanism",
        "Cholesky mechanism",
    ]
    method_codes = ["euclidean_gaussian_ar", "cholesky_gaussian_ar", "log_euclidean_gaussian_ar", "bwar_barycenter"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Parameter and generating-coordinate variations for the Gaussian simulation. Entries are full \(W_2^2\) loss ratios to persistence, reported as mean (standard error) over {reps if reps is not None else 50} replications. The first five rows use the Bures transport generator; the final two rows use Log-Euclidean and log-diagonal Cholesky covariance coordinates, respectively, with {ar_model} ridge VAR(1) lag-design fits. Lower values are better.}}",
        r"\label{tab:synthetic-transport-variation}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Design & \(T\) & \(d\) & \(\phi\) & Dispersion & Euclidean AR & Cholesky AR & Log-Euclidean AR & BWAR \\",
        r"\midrule",
    ]
    for design in designs:
        part = summary.loc[summary["design"].eq(design)]
        first = part.iloc[0]
        stats = {row["method"]: row for _, row in part.iterrows()}
        best = min(float(stats[m]["w2_ratio_mean"]) for m in method_codes)
        cells = [
            fmt_mean_se(float(stats[m]["w2_ratio_mean"]), float(stats[m]["w2_ratio_se"]), best=best)
            for m in method_codes
        ]
        lines.append(
            f"{tex_escape(design)} & {int(first['n'])} & {int(first['d'])} & "
            f"{float(first['phi']):.2f} & {float(first['dispersion']):.2f} & "
            + " & ".join(cells)
            + r" \\"
        )
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


def make_figure(raw: pd.DataFrame, summary: pd.DataFrame, path_stem: Path) -> None:
    plt = import_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.36)
    ax_rep = fig.add_subplot(gs[0, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[1, :])

    baseline = raw.loc[raw["design"].eq("Baseline") & raw["method"].isin(PLOT_METHODS)].copy()
    positions = np.arange(len(PLOT_METHODS))
    rng = np.random.default_rng(20260707)
    for i, method in enumerate(PLOT_METHODS):
        vals = baseline.loc[baseline["method"].eq(method), "w2_ratio_to_persistence"].astype(float).to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_rep.scatter(
            np.full(len(vals), positions[i]) + jitter,
            vals,
            s=14,
            color=METHOD_COLOR[method],
            alpha=0.62,
            linewidth=0.0,
        )
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax_rep.plot([positions[i] - 0.19, positions[i] + 0.19], [med, med], color="#111827", lw=1.2)
        ax_rep.add_patch(
            plt.Rectangle(
                (positions[i] - 0.16, q1),
                0.32,
                q3 - q1,
                fill=False,
                edgecolor="#111827",
                lw=0.8,
            )
        )
    ax_rep.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--")
    ax_rep.set_xticks(positions)
    ax_rep.set_xticklabels([METHOD_LABEL[m] for m in PLOT_METHODS])
    ax_rep.set_ylabel(r"Full $W_2^2$ ratio")
    ax_rep.set_title("Replicate-level main setting", loc="left", fontweight="bold")
    ax_rep.grid(axis="y", color="#E5E7EB", lw=0.55)

    cov_part = summary.loc[summary["design"].eq("Baseline") & summary["method"].isin(PLOT_METHODS)].copy()
    cov_part["order"] = cov_part["method"].map({m: i for i, m in enumerate(PLOT_METHODS)})
    cov_part = cov_part.sort_values("order")
    ax_cov.bar(
        np.arange(len(cov_part)),
        cov_part["cov_ratio_mean"].astype(float),
        yerr=cov_part["cov_ratio_se"].astype(float),
        color=[METHOD_COLOR[m] for m in cov_part["method"]],
        edgecolor="#263445",
        linewidth=0.55,
        capsize=2.2,
        width=0.62,
    )
    ax_cov.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--")
    ax_cov.set_xticks(np.arange(len(cov_part)))
    ax_cov.set_xticklabels([METHOD_LABEL[m] for m in cov_part["method"]])
    ax_cov.set_ylabel("Covariance loss ratio")
    ax_cov.set_title("Covariance component", loc="left", fontweight="bold")
    ax_cov.grid(axis="y", color="#E5E7EB", lw=0.55)

    designs = ["Baseline", "Shorter series", "Higher dimension", "Weaker dynamics", "Larger variation"]
    x = np.arange(len(designs))
    offsets = np.linspace(-0.27, 0.27, len(PLOT_METHODS))
    for off, method in zip(offsets, PLOT_METHODS):
        part = summary.loc[
            summary["method"].eq(method) & summary["design"].isin(designs)
        ].copy()
        part["design"] = pd.Categorical(part["design"], categories=designs, ordered=True)
        part = part.sort_values("design")
        ax_var.errorbar(
            x + off,
            part["w2_ratio_mean"].astype(float),
            yerr=part["w2_ratio_se"].astype(float),
            marker="o",
            lw=1.4 if method != "bwar_barycenter" else 2.0,
            ms=4.0,
            capsize=2.0,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
        )
    ax_var.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--")
    ax_var.set_xticks(x)
    ax_var.set_xticklabels(["Baseline", "Shorter\nseries", "Higher\ndimension", "Weaker\ndynamics", "Larger\nvariation"])
    ax_var.set_ylabel(r"Full $W_2^2$ ratio")
    ax_var.set_title("Parameter variation", loc="left", fontweight="bold")
    ax_var.grid(axis="y", color="#E5E7EB", lw=0.55)
    ax_var.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, handlelength=1.8, columnspacing=1.1)

    for label, ax in zip("abc", [ax_rep, ax_cov, ax_var]):
        panel_label(ax, label)
    fig.subplots_adjust(bottom=0.16, top=0.94)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=420, bbox_inches="tight")
    plt.close(fig)


def run(reps: int, *, ar_model: str = "diag") -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    settings = default_settings()
    for setting in settings:
        print(
            f"[synthetic] {setting['design']}: n={setting['n']} d={setting['d']} "
            f"phi={setting['phi']} reps={reps}",
            flush=True,
        )
        for seed in range(reps):
            frames.append(run_setting(seed=seed, ar_model=ar_model, **setting))
    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw)
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strong fixed-barycenter synthetic artifacts.")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, summary = run(args.reps, ar_model=args.ar_model)
    raw.to_csv(OUT_DIR / "strong_synthetic_transport_raw.csv", index=False)
    summary.to_csv(OUT_DIR / "strong_synthetic_transport_summary.csv", index=False)
    write_main_table(summary, TABLE_DIR / "synthetic_transport_main.tex")
    write_variation_table(summary, TABLE_DIR / "synthetic_transport_variation.tex")
    make_figure(raw, summary, FIGURE_DIR / "synthetic_transport_mechanism")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
