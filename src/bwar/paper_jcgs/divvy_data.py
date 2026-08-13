"""Download and preprocess the Divvy trip records used in the article."""

from __future__ import annotations

from pathlib import Path
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from bwar.gaussian_geometry import project_spd
from bwar.paper_jcgs.rolling_origin import make_rolling_origin_splits


ROOT = Path(__file__).resolve().parents[3]
DIVVY_URL = "https://divvy-tripdata.s3.amazonaws.com/{month}-divvy-tripdata.zip"


def divvy_zip_path(month: str) -> Path:
    return download_url(
        DIVVY_URL.format(month=month),
        ROOT / "data" / "divvy" / f"{month}.zip",
    )


def load_divvy_hourly(months: tuple[str, ...]) -> pd.DataFrame:
    cache_path = ROOT / "data" / "divvy" / f"hourly_{'_'.join(months)}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    monthly = [_read_monthly_station_counts(divvy_zip_path(month)) for month in months]
    counts = monthly[0]
    for part in monthly[1:]:
        counts = counts.add(part, fill_value=0.0)
    frame = counts.unstack(fill_value=0.0).sort_index()
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index, fill_value=0.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    return frame


def divvy_matrix(
    n_stations: int,
    *,
    months: tuple[str, ...],
    window: int | None = None,
    step: int | None = None,
    max_matrices: int | None = None,
    fit_raw_end: int | None = None,
    return_columns: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[str, ...]]:
    frame = load_divvy_hourly(months)
    early_end = fit_raw_end
    if early_end is None and window is not None and step is not None and max_matrices is not None:
        early_end = _initial_fit_raw_end(
            n_raw=len(frame),
            window=window,
            step=step,
            max_matrices=max_matrices,
        )
    values = frame.clip(lower=0.0).to_numpy(float)
    selected = _select_variable_indices(values, n_stations, early_end=early_end)
    matrix = values[:, selected]
    if return_columns:
        return matrix, tuple(str(frame.columns[index]) for index in selected)
    return matrix


def download_url(url: str, path: Path, *, timeout: int = 240) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        path.write_bytes(resp.read())
    return path


def _read_monthly_station_counts(path: Path) -> pd.Series:
    counts: pd.Series | None = None
    with zipfile.ZipFile(path) as zf:
        csv_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv")
            and "__macosx/" not in name.lower()
            and not Path(name).name.startswith("._")
        ]
        if not csv_names:
            raise ValueError(f"no CSV file in {path}")
        for csv_name in csv_names:
            with zf.open(csv_name) as fh:
                header = pd.read_csv(fh, nrows=0)
            if "started_at" not in header.columns:
                continue
            station_col = "start_station_id" if "start_station_id" in header.columns else "start_station_name"
            if station_col not in header.columns:
                continue
            usecols = ["started_at", station_col]
            with zf.open(csv_name) as fh:
                for chunk in pd.read_csv(fh, usecols=usecols, chunksize=350_000):
                    chunk = chunk.dropna(subset=["started_at", station_col]).copy()
                    if chunk.empty:
                        continue
                    chunk["hour"] = pd.to_datetime(chunk["started_at"], errors="coerce").dt.floor("h")
                    chunk = chunk.dropna(subset=["hour"])
                    chunk[station_col] = chunk[station_col].astype(str)
                    part = chunk.groupby(["hour", station_col], observed=True).size()
                    counts = part if counts is None else counts.add(part, fill_value=0.0)
    if counts is None:
        return pd.Series(dtype=float, name="count")
    counts.name = "count"
    return counts.astype(float)


def rolling_gaussians_from_array(
    X: np.ndarray,
    *,
    window: int,
    step: int,
    max_matrices: int = 900,
    ridge: float = 1e-5,
    standardize: bool = True,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if standardize:
        scale_end = max(window, int(0.45 * len(X)))
        mu = X[:scale_end].mean(axis=0)
        sd = X[:scale_end].std(axis=0)
        sd[sd < 1e-8] = 1.0
        X = (X - mu) / sd
    starts = list(range(0, len(X) - window + 1, step))
    if not starts:
        d = X.shape[1]
        if return_windows:
            return np.empty((0, d)), np.empty((0, d, d)), np.empty((0, window, d)), np.empty((0,), dtype=int)
        return np.empty((0, d)), np.empty((0, d, d))
    if len(starts) > max_matrices:
        idx = np.linspace(0, len(starts) - 1, max_matrices, dtype=int)
        starts = [starts[i] for i in idx]
    means = []
    covs = []
    windows = []
    for start in starts:
        W = X[start : start + window]
        C = np.cov(W, rowvar=False)
        C = np.atleast_2d(C)
        means.append(W.mean(axis=0))
        covs.append(project_spd(C + ridge * np.eye(C.shape[0]), eps=1e-8))
        if return_windows:
            windows.append(W.copy())
    if return_windows:
        return np.asarray(means), np.asarray(covs), np.asarray(windows), np.asarray(starts, dtype=int)
    return np.asarray(means), np.asarray(covs)


def _initial_fit_raw_end(*, n_raw: int, window: int, step: int, max_matrices: int) -> int:
    starts = list(range(0, n_raw - window + 1, step))
    if len(starts) > max_matrices:
        idx = np.linspace(0, len(starts) - 1, max_matrices, dtype=int)
        starts = [starts[i] for i in idx]
    splits = make_rolling_origin_splits(len(starts), max_origins=1)
    if not splits:
        return min(n_raw, max(window, 1))
    fit_end = splits[0][0]
    return int(starts[fit_end - 1] + window)


def _standardized_stream_with_profile(
    X: np.ndarray,
    *,
    window: int,
    step: int,
    max_matrices: int,
    metric: str,
    label: str,
    fit_raw_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X = np.asarray(X, dtype=float)
    X = X[:, np.nanstd(X, axis=0) > 1e-10]
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("raw physical matrix has no usable columns")
    X_filled = pd.DataFrame(X).interpolate(limit_direction="both").dropna(axis=0, how="any").to_numpy(float)
    scale_end = (
        _initial_fit_raw_end(
            n_raw=len(X_filled),
            window=window,
            step=step,
            max_matrices=max_matrices,
        )
        if fit_raw_end is None
        else min(len(X_filled), max(1, int(fit_raw_end)))
    )
    center = X_filled[:scale_end].mean(axis=0)
    scale = X_filled[:scale_end].std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    X_standardized = (X_filled - center) / scale
    means, covs, raw_windows, starts = rolling_gaussians_from_array(
        X_standardized,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=False,
        return_windows=True,
    )
    profile = {
        "metric": metric,
        "label": label,
        "kind": "mean_rmse",
        "center": center[: means.shape[1]],
        "scale": scale[: means.shape[1]],
        "fit_raw_end": int(scale_end),
    }
    return means, np.asarray([project_spd(C, eps=1e-8) for C in covs]), raw_windows, starts, profile


def _select_variable_indices(X: np.ndarray, n_features: int, *, early_end: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < n_features:
        raise ValueError(f"matrix has shape {X.shape}, cannot select {n_features} columns")
    if early_end is None:
        early_end = max(10, int(0.45 * len(X)))
    early_end = min(len(X), max(1, int(early_end)))
    early = X[:early_end]
    nonzero = np.nanmean(early > 0.0, axis=0)
    variability = np.nanstd(early, axis=0)
    scores = np.nan_to_num(nonzero * variability, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    return np.argsort(scores)[::-1][:n_features]


def _select_variable_columns(X: np.ndarray, n_features: int, *, early_end: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    selected = _select_variable_indices(X, n_features, early_end=early_end)
    return X[:, selected]
