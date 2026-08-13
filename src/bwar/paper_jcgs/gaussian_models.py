"""Gaussian encodings, losses, and autoregressive forecasting models."""

from __future__ import annotations

import numpy as np

from bwar.gaussian_geometry import (
    bw2_cov,
    bw_barycenter,
    mat_exp,
    mat_from_triu,
    mat_log,
    ot_map,
    project_spd,
    triu_vec,
)


DEFAULT_RIDGE_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
REFERENCE_LIBRARY_MODE = "full"
DEFAULT_DOMAIN_METRIC_PROFILE = {
    "metric": "station_demand_level_rmse",
    "label": "training-standardized station-demand mean RMSE",
    "kind": "mean_rmse",
}


def domain_metric_profile(dataset: str | None) -> dict[str, object]:
    """Return the loss profile used by the Divvy analysis."""
    del dataset
    return dict(DEFAULT_DOMAIN_METRIC_PROFILE)


def domain_loss_from_moments(
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    profile: dict[str, object] | None = None,
) -> float:
    """Compute the standardized station-demand mean RMSE."""
    del pred_cov, target_cov
    profile = dict(DEFAULT_DOMAIN_METRIC_PROFILE if profile is None else profile)
    if profile.get("kind", "mean_rmse") != "mean_rmse":
        raise ValueError("the public Divvy package supports only mean_rmse")
    pred_mean = np.asarray(pred_mean, dtype=float)
    target_mean = np.asarray(target_mean, dtype=float)
    return float(np.sqrt(np.mean((pred_mean - target_mean) ** 2)))


def metric_mean_key(metric: str, prefix: str) -> str:
    if metric in {"domain", "domain_loss"}:
        return f"{prefix}_domain_loss_mean"
    if metric == "w2":
        return f"{prefix}_w2_mean"
    if metric == "log_score":
        return f"{prefix}_log_score_mean"
    raise ValueError(f"unknown metric: {metric}")


def gaussian_w2_squared(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    return float(np.sum((np.asarray(mean_a) - np.asarray(mean_b)) ** 2) + bw2_cov(cov_a, cov_b))


def gaussian_log_score_from_moments(
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """Average Gaussian negative log likelihood of a target window from its moments."""
    pred_mean = np.asarray(pred_mean, dtype=float)
    target_mean = np.asarray(target_mean, dtype=float)
    pred_cov = project_spd(np.asarray(pred_cov, dtype=float), eps=eps)
    target_cov = project_spd(np.asarray(target_cov, dtype=float), eps=eps)
    sign, logdet = np.linalg.slogdet(pred_cov)
    if sign <= 0:
        pred_cov = project_spd(pred_cov, eps=max(eps, 1e-6))
        sign, logdet = np.linalg.slogdet(pred_cov)
    diff = target_mean - pred_mean
    solved_target = np.linalg.solve(pred_cov, target_cov)
    solved_diff = np.linalg.solve(pred_cov, diff)
    d = pred_cov.shape[0]
    return float(
        0.5
        * (
            d * np.log(2.0 * np.pi)
            + logdet
            + np.trace(solved_target)
            + float(diff @ solved_diff)
        )
    )


def safe_transport_from_vech(z: np.ndarray, d: int, min_eig: float = 0.05) -> np.ndarray:
    H = mat_from_triu(z, d)
    vals = np.linalg.eigvalsh(np.eye(d) + H)
    if vals.min() < min_eig:
        h_min = float(np.linalg.eigvalsh(H).min())
        if h_min < 0.0:
            scale = (1.0 - min_eig) / max(abs(h_min), 1e-12)
            H = scale * H
    return project_spd(np.eye(d) + H, eps=1e-8)


def bwar_gaussian_encode(
    mean: np.ndarray,
    cov: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
) -> np.ndarray:
    d = ref_cov.shape[0]
    A = ot_map(ref_cov, cov)
    return np.r_[np.asarray(mean) - np.asarray(ref_mean), triu_vec(A - np.eye(d))]


def bwar_gaussian_decode(
    z: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = ref_cov.shape[0]
    pred_mean = np.asarray(ref_mean) + np.asarray(z[:d])
    A = safe_transport_from_vech(np.asarray(z[d:]), d)
    pred_cov = project_spd(A @ ref_cov @ A, eps=1e-8)
    return pred_mean, pred_cov


def cholesky_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = project_spd(cov, eps=1e-8)
    L = np.linalg.cholesky(cov)
    idx = np.tril_indices(cov.shape[0])
    z = L[idx].copy()
    diag_positions = [np.where((idx[0] == j) & (idx[1] == j))[0][0] for j in range(cov.shape[0])]
    z[diag_positions] = np.log(np.clip(z[diag_positions], 1e-12, None))
    return np.r_[mean, z]


def cholesky_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(z[:d])
    lower = np.asarray(z[d:])
    L = np.zeros((d, d), dtype=float)
    idx = np.tril_indices(d)
    L[idx] = lower
    for j in range(d):
        pos = np.where((idx[0] == j) & (idx[1] == j))[0][0]
        L[j, j] = np.exp(np.clip(L[j, j], -30, 30))
    return mean, project_spd(L @ L.T, eps=1e-8)


def euclidean_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.r_[mean, triu_vec(cov)]


def euclidean_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[:d]), project_spd(mat_from_triu(np.asarray(z[d:]), d), eps=1e-8)


def log_euclidean_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.r_[mean, triu_vec(mat_log(cov))]


def log_euclidean_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[:d]), project_spd(mat_exp(mat_from_triu(np.asarray(z[d:]), d)), eps=1e-8)


def fit_var(Z: np.ndarray, end: int, *, lam: float, model: str) -> np.ndarray:
    """Fit the article's ridge lag-design VAR(1) in row-vector form.

    This is the ``p=1`` special case of equation (3.8): coordinates are
    centered by the fitting-block mean, the lagged design is formed from
    ``Z[:-1]``, and ridge is applied to its positive-semidefinite Gram matrix.
    ``model='diag'`` applies the same normal equations coordinate-wise;
    ``model='full'`` fits the unrestricted VAR(1).  The returned augmented
    matrix stores the row coefficient matrix below the intercept, so recursive
    forecasts implement ``mean + (state - mean) @ A``.
    """

    fitted = np.asarray(Z[:end], dtype=float)
    if len(fitted) < 3:
        raise ValueError("not enough observations to fit VAR")
    if not np.isfinite(fitted).all():
        raise ValueError("VAR coordinates must be finite")
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a finite positive scalar")

    n, q = fitted.shape
    coordinate_mean = fitted.mean(axis=0)
    centered = fitted - coordinate_mean
    lagged = centered[:-1]
    response = centered[1:]
    n_lagged = len(lagged)

    if model == "diag":
        gram_diagonal = np.sum(lagged**2, axis=0)
        cross_diagonal = np.sum(lagged * response, axis=0)
        row_coefficients = np.diag(
            cross_diagonal / (gram_diagonal + n_lagged * lam)
        )
    elif model == "full":
        gram = lagged.T @ lagged
        cross = lagged.T @ response
        row_coefficients = np.linalg.solve(
            gram + n_lagged * lam * np.eye(q),
            cross,
        )
    else:
        raise ValueError(f"unknown ar_model: {model}")

    W = np.empty((q + 1, q), dtype=float)
    W[0] = coordinate_mean - coordinate_mean @ row_coefficients
    W[1:] = row_coefficients
    return W


def recursive_predict_z(z0: np.ndarray, W: np.ndarray, horizon: int) -> np.ndarray:
    z = np.asarray(z0, dtype=float)
    for _ in range(horizon):
        z = np.r_[1.0, z] @ W
    return z


def score_recursive_forecasts(
    means: np.ndarray,
    covs: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    decode,
    *,
    start_t: int,
    stop_t: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    losses = []
    log_scores = []
    domain_losses = []
    min_eigs = []
    for t in range(start_t, stop_t):
        if t + horizon >= len(covs):
            continue
        pred_mean, pred_cov = decode(recursive_predict_z(Z[t], W, horizon))
        pred_cov = project_spd(pred_cov, eps=1e-8)
        losses.append(gaussian_w2_squared(pred_mean, pred_cov, means[t + horizon], covs[t + horizon]))
        log_scores.append(
            gaussian_log_score_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon])
        )
        domain_losses.append(
            domain_loss_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon], profile)
        )
        min_eigs.append(float(np.linalg.eigvalsh(pred_cov).min()))
    arr = np.asarray(losses, dtype=float)
    log_arr = np.asarray(log_scores, dtype=float)
    domain_arr = np.asarray(domain_losses, dtype=float)
    if len(arr) == 0:
        return {
            "w2_mean": np.nan,
            "w2_median": np.nan,
            "w2_q90": np.nan,
            "log_score_mean": np.nan,
            "log_score_median": np.nan,
            "log_score_q90": np.nan,
            "domain_loss_mean": np.nan,
            "domain_loss_median": np.nan,
            "domain_loss_q90": np.nan,
            "n_pairs": 0,
            "min_pred_eig": np.nan,
        }
    return {
        "w2_mean": float(arr.mean()),
        "w2_median": float(np.median(arr)),
        "w2_q90": float(np.quantile(arr, 0.9)),
        "log_score_mean": float(log_arr.mean()),
        "log_score_median": float(np.median(log_arr)),
        "log_score_q90": float(np.quantile(log_arr, 0.9)),
        "domain_loss_mean": float(domain_arr.mean()),
        "domain_loss_median": float(np.median(domain_arr)),
        "domain_loss_q90": float(np.quantile(domain_arr, 0.9)),
        "n_pairs": int(len(arr)),
        "min_pred_eig": float(np.min(min_eigs)),
    }


def persistence_metrics(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile

    def score(start_t: int, stop_t: int) -> dict[str, float | int]:
        losses = []
        log_scores = []
        domain_losses = []
        for t in range(start_t, stop_t):
            if t + horizon < len(covs):
                losses.append(gaussian_w2_squared(means[t], covs[t], means[t + horizon], covs[t + horizon]))
                log_scores.append(
                    gaussian_log_score_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon])
                )
                domain_losses.append(
                    domain_loss_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon], profile)
                )
        arr = np.asarray(losses, dtype=float)
        log_arr = np.asarray(log_scores, dtype=float)
        domain_arr = np.asarray(domain_losses, dtype=float)
        return {
            "w2_mean": float(arr.mean()) if len(arr) else np.nan,
            "w2_median": float(np.median(arr)) if len(arr) else np.nan,
            "w2_q90": float(np.quantile(arr, 0.9)) if len(arr) else np.nan,
            "log_score_mean": float(log_arr.mean()) if len(log_arr) else np.nan,
            "log_score_median": float(np.median(log_arr)) if len(log_arr) else np.nan,
            "log_score_q90": float(np.quantile(log_arr, 0.9)) if len(log_arr) else np.nan,
            "domain_loss_mean": float(domain_arr.mean()) if len(domain_arr) else np.nan,
            "domain_loss_median": float(np.median(domain_arr)) if len(domain_arr) else np.nan,
            "domain_loss_q90": float(np.quantile(domain_arr, 0.9)) if len(domain_arr) else np.nan,
            "n_pairs": int(len(arr)),
        }

    val = score(max(0, fit_end - horizon), max(0, val_end - horizon))
    test = score(max(0, val_end - horizon), len(covs) - horizon)
    return {
        "ridge": np.nan,
        "val_w2_mean": float(val["w2_mean"]),
        "val_w2_median": float(val["w2_median"]),
        "val_log_score_mean": float(val["log_score_mean"]),
        "val_log_score_median": float(val["log_score_median"]),
        "val_domain_loss_mean": float(val["domain_loss_mean"]),
        "val_domain_loss_median": float(val["domain_loss_median"]),
        "test_w2_mean": float(test["w2_mean"]),
        "test_w2_median": float(test["w2_median"]),
        "test_w2_q90": float(test["w2_q90"]),
        "test_log_score_mean": float(test["log_score_mean"]),
        "test_log_score_median": float(test["log_score_median"]),
        "test_log_score_q90": float(test["log_score_q90"]),
        "test_domain_loss_mean": float(test["domain_loss_mean"]),
        "test_domain_loss_median": float(test["domain_loss_median"]),
        "test_domain_loss_q90": float(test["domain_loss_q90"]),
        "n_test_pairs": int(test["n_pairs"]),
        "min_pred_eig": float(min(np.linalg.eigvalsh(C).min() for C in covs[max(0, val_end - horizon) : len(covs) - horizon])),
    }


def bw_geodesic(S0: np.ndarray, S1: np.ndarray, alpha: float) -> np.ndarray:
    A = ot_map(S0, S1)
    M = (1.0 - alpha) * np.eye(S0.shape[0]) + alpha * A
    return project_spd(M @ S0 @ M, eps=1e-8)


def candidate_gaussian_references(
    means_fit: np.ndarray,
    covs_fit: np.ndarray,
    *,
    max_sample_refs: int = 4,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    train_covs = np.asarray([project_spd(C, eps=1e-8) for C in covs_fit])
    train_means = np.asarray(means_fit, dtype=float)
    d = train_covs.shape[1]
    mean_ref = train_means.mean(axis=0)
    pooled = project_spd(np.mean(train_covs, axis=0), eps=1e-8)
    diag_pooled = project_spd(np.diag(np.diag(pooled)), eps=1e-8)
    scaled_identity = project_spd(np.eye(d) * np.trace(pooled) / d, eps=1e-8)
    refs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("pooled_cov", mean_ref, pooled),
        ("log_euclidean_mean", mean_ref, mat_exp(np.mean([mat_log(C) for C in train_covs], axis=0))),
        ("diag_pooled", mean_ref, diag_pooled),
        ("scaled_identity", mean_ref, scaled_identity),
    ]
    if REFERENCE_LIBRARY_MODE == "fast":
        refs.append(("geodesic_identity_a0.25", mean_ref, bw_geodesic(pooled, scaled_identity, 0.25)))
        return refs
    if REFERENCE_LIBRARY_MODE == "full":
        try:
            refs.append(("bw_barycenter", mean_ref, bw_barycenter(train_covs[: min(len(train_covs), 250)])))
        except Exception:
            pass

    idx = np.linspace(0, len(train_covs) - 1, min(max_sample_refs, len(train_covs)), dtype=int)
    for j, i in enumerate(idx):
        refs.append((f"sample_{j}", train_means[i], train_covs[i]))
    for alpha in (0.25, 0.5, 0.75):
        refs.append((f"geodesic_diag_a{alpha}", mean_ref, bw_geodesic(pooled, diag_pooled, alpha)))
        refs.append((f"geodesic_identity_a{alpha}", mean_ref, bw_geodesic(pooled, scaled_identity, alpha)))
    return refs
