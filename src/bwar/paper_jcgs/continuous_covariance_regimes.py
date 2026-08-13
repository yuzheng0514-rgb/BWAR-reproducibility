"""Continuous covariance regimes for the Fixed-versus-Local BWAR study.

Both regimes generate time-varying Gaussian covariances at every stage of the
chronological split.  They differ in the low-frequency structure of that
variation: a persistent directional component versus stationary mean
reversion around an invariant center.  No change point is inserted at the fit,
validation, or test boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from bwar.gaussian_geometry import (
    bw2_cov,
    mat_exp,
    mat_from_triu,
    ot_map,
    project_spd,
)
from bwar.paper_jcgs.gaussian_models import (
    cholesky_decode,
    cholesky_encode,
    euclidean_encode,
    fit_var,
    log_euclidean_encode,
    recursive_predict_z,
)
from bwar.paper_jcgs.local_reference_bwar import (
    build_local_bwar_geometry,
    exact_bures_barycenter,
    local_bwar_decode,
    local_bwar_encode,
)


REGIMES = ("persistent", "mean_reverting")
METHODS = (
    "persistence",
    "euclidean",
    "cholesky",
    "log_euclidean",
    "fixed",
    "local",
)


@dataclass(frozen=True)
class ContinuousCovarianceConfig:
    """Configuration shared by the two continuous covariance processes."""

    n: int = 360
    d: int = 3
    fit_end: int = 160
    val_end: int = 240
    window_length: int = 24
    phi: float = 0.55
    ar_model: str = "full"
    target_deltas: tuple[float, ...] = (0.0, 0.75, 1.50, 2.00)
    horizons: tuple[int, ...] = (1, 3, 6)
    ridge_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    seed_base: int = 15_000_000
    k_ref: int = 3
    refresh_period: int = 10
    reference_condition: float = 12.0
    covariance_nugget: float = 0.5
    cyclic_period: int = 15
    cyclic_amplitudes: tuple[float, ...] = (0.84, 0.672, 0.504)
    cyclic_noise_dispersion: float = 0.002
    mean_reversion_controls: tuple[float, ...] = (0.40, 0.60, 0.80, 1.00)
    fixed_center_mean_cycle_scale: float = 0.0
    fixed_center_mean_cycle_period: int = 30
    fixed_center_cyclic_period: int = 30
    mean_dispersion: float = 0.015
    regimes: tuple[str, ...] = REGIMES
    residual_threshold: float = 1e-4
    minimum_generating_transport_eigenvalue: float = 0.12
    max_seed_resamples: int = 20
    seed_retry_stride: int = 10_000_000

    def validate(self) -> None:
        if self.d < 2 or self.n < 60:
            raise ValueError("invalid dimension or series length")
        if self.ar_model != "full":
            raise ValueError("this study requires a full VAR")
        if not (self.window_length < self.fit_end < self.val_end < self.n):
            raise ValueError("invalid window/fit/validation boundaries")
        if max(self.horizons) >= min(
            self.val_end - self.fit_end,
            self.n - self.val_end,
        ):
            raise ValueError("blocks are too short for requested horizons")
        if not self.target_deltas or not np.isclose(min(self.target_deltas), 0.0):
            raise ValueError("target_deltas must include zero")
        if any(
            right <= left
            for left, right in zip(self.target_deltas, self.target_deltas[1:])
        ):
            raise ValueError("target_deltas must be strictly increasing")
        if self.k_ref < 1 or self.refresh_period < 1:
            raise ValueError("invalid local-reference controls")
        if not 0.0 < self.minimum_generating_transport_eigenvalue < 1.0:
            raise ValueError("invalid generating-transport floor")
        if self.max_seed_resamples < 0 or self.seed_retry_stride < 1:
            raise ValueError("invalid seed-rejection controls")
        if len(self.cyclic_amplitudes) != self.d:
            raise ValueError("one cyclic amplitude is required per dimension")
        if not self.mean_reversion_controls or any(
            right <= left
            for left, right in zip(
                self.mean_reversion_controls,
                self.mean_reversion_controls[1:],
            )
        ):
            raise ValueError("mean-reversion controls must be strictly increasing")
        if self.fixed_center_cyclic_period < 2:
            raise ValueError("fixed-center covariance period must be at least two")
        if self.fixed_center_mean_cycle_period < 2:
            raise ValueError("fixed-center mean period must be at least two")
        if self.fixed_center_mean_cycle_scale < 0.0:
            raise ValueError("fixed-center mean cycle scale must be nonnegative")
        if tuple(self.regimes) != REGIMES:
            raise ValueError(f"regimes must equal {REGIMES}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ContinuousCovarianceConfig":
        config = cls(
            n=int(values.get("n", cls.n)),
            d=int(values.get("d", cls.d)),
            fit_end=int(values.get("fit_end", cls.fit_end)),
            val_end=int(values.get("val_end", cls.val_end)),
            window_length=int(values.get("window_length", cls.window_length)),
            phi=float(values.get("phi", cls.phi)),
            ar_model=str(values.get("ar_model", cls.ar_model)),
            target_deltas=tuple(float(v) for v in values.get("target_deltas", cls.target_deltas)),
            horizons=tuple(int(v) for v in values.get("horizons", cls.horizons)),
            ridge_grid=tuple(float(v) for v in values.get("ridge_grid", cls.ridge_grid)),
            seed_base=int(values.get("seed_base", cls.seed_base)),
            k_ref=int(values.get("k_ref", cls.k_ref)),
            refresh_period=int(values.get("refresh_period", cls.refresh_period)),
            reference_condition=float(values.get("reference_condition", cls.reference_condition)),
            covariance_nugget=float(values.get("covariance_nugget", cls.covariance_nugget)),
            cyclic_period=int(values.get("cyclic_period", cls.cyclic_period)),
            cyclic_amplitudes=tuple(float(v) for v in values.get("cyclic_amplitudes", cls.cyclic_amplitudes)),
            cyclic_noise_dispersion=float(values.get("cyclic_noise_dispersion", cls.cyclic_noise_dispersion)),
            mean_reversion_controls=tuple(float(v) for v in values.get("mean_reversion_controls", cls.mean_reversion_controls)),
            fixed_center_mean_cycle_scale=float(values.get("fixed_center_mean_cycle_scale", cls.fixed_center_mean_cycle_scale)),
            fixed_center_cyclic_period=int(values.get("fixed_center_cyclic_period", cls.fixed_center_cyclic_period)),
            fixed_center_mean_cycle_period=int(values.get("fixed_center_mean_cycle_period", cls.fixed_center_mean_cycle_period)),
            mean_dispersion=float(values.get("mean_dispersion", cls.mean_dispersion)),
            regimes=tuple(str(v) for v in values.get("regimes", cls.regimes)),
            residual_threshold=float(values.get("residual_threshold", cls.residual_threshold)),
            minimum_generating_transport_eigenvalue=float(values.get("minimum_generating_transport_eigenvalue", cls.minimum_generating_transport_eigenvalue)),
            max_seed_resamples=int(values.get("max_seed_resamples", cls.max_seed_resamples)),
            seed_retry_stride=int(values.get("seed_retry_stride", cls.seed_retry_stride)),
        )
        config.validate()
        return config


def _spectral_normalize(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    return symmetric / max(float(np.abs(np.linalg.eigvalsh(symmetric)).max()), 1e-12)


def _directions(base_covariance: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    _, eigenvectors = np.linalg.eigh(base_covariance)
    d = base_covariance.shape[0]
    persistent_basis = np.zeros((d, d), dtype=float)
    for index in range(d - 1):
        persistent_basis[index, index + 1] = 1.0 - 0.12 * index
        persistent_basis[index + 1, index] = 1.0 - 0.12 * index
    persistent_direction = eigenvectors @ _spectral_normalize(persistent_basis) @ eigenvectors.T
    local_directions = []
    for index in range(d):
        basis = np.zeros((d, d), dtype=float)
        basis[index, index] = 1.0
        local_directions.append(eigenvectors @ basis @ eigenvectors.T)
    return persistent_direction, tuple(local_directions)


@dataclass(frozen=True)
class _MethodSpec:
    name: str
    encode: Callable[[np.ndarray, np.ndarray], np.ndarray]
    decode: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, float, bool]]


def _method_specs(
    *,
    d: int,
    projection_eig: float = 1e-8,
) -> tuple[_MethodSpec, ...]:
    """Construct the three baseline covariance-coordinate charts."""

    def euclidean_decode(coordinate: np.ndarray):
        raw_covariance = mat_from_triu(np.asarray(coordinate[d:]), d)
        raw_minimum = float(np.linalg.eigvalsh(raw_covariance).min())
        return (
            np.asarray(coordinate[:d]),
            project_spd(raw_covariance, eps=projection_eig),
            raw_minimum,
            raw_minimum < projection_eig,
        )

    def cholesky_decode_instrumented(coordinate: np.ndarray):
        mean, covariance = cholesky_decode(coordinate, d)
        return mean, covariance, float(np.linalg.eigvalsh(covariance).min()), False

    def log_decode(coordinate: np.ndarray):
        mean = np.asarray(coordinate[:d])
        covariance = mat_exp(mat_from_triu(np.asarray(coordinate[d:]), d))
        return mean, covariance, float(np.linalg.eigvalsh(covariance).min()), False

    return (
        _MethodSpec("euclidean", euclidean_encode, euclidean_decode),
        _MethodSpec("cholesky", cholesky_encode, cholesky_decode_instrumented),
        _MethodSpec("log_euclidean", log_euclidean_encode, log_decode),
    )


def _calibrate_multiplier(
    *,
    target_delta: float,
    base_covariance: np.ndarray,
    drift_direction: np.ndarray,
) -> float:
    """Calibrate an exponential transport path to a Bures endpoint distance."""

    target = float(target_delta)
    if target == 0.0:
        return 0.0

    def distance(multiplier: float) -> float:
        deformation = mat_exp(multiplier * drift_direction)
        shifted = deformation @ base_covariance @ deformation
        return float(np.sqrt(max(bw2_cov(base_covariance, shifted), 0.0)))

    lower, upper = 0.0, 1.0
    while distance(upper) < target and upper < 128.0:
        upper *= 2.0
    if distance(upper) < target:
        raise RuntimeError("target displacement exceeds calibration range")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if distance(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _max_commutator_reference(
    d: int,
    *,
    condition: float,
    amplitudes: np.ndarray,
    nugget: float,
) -> np.ndarray:
    """Deterministic reference whose spectrum maximizes cyclic misalignment."""

    from itertools import permutations

    rows = np.arange(d, dtype=float)[:, None]
    columns = np.arange(d, dtype=float)[None, :]
    eigenvectors = np.sqrt(2.0 / d) * np.cos(
        np.pi * (rows + 0.5) * columns / d
    )
    eigenvectors[:, 0] /= np.sqrt(2.0)
    eigenvalues = np.exp(
        np.linspace(-0.5 * np.log(condition), 0.5 * np.log(condition), d)
    )
    direction = np.diag(amplitudes)

    def score(order: tuple[float, ...]) -> float:
        covariance = (eigenvectors * np.asarray(order)) @ eigenvectors.T
        commutator = covariance @ direction - direction @ covariance
        return float(np.linalg.norm(commutator, "fro"))

    ordered = np.asarray(
        max(permutations(eigenvalues.tolist()), key=score),
        dtype=float,
    )
    covariance = (eigenvectors * ordered) @ eigenvectors.T
    return project_spd(covariance + nugget * np.eye(d), eps=1e-8)


def _stationary_ar(
    rng: np.random.Generator,
    *,
    n: int,
    phis: np.ndarray,
    scale: float,
) -> np.ndarray:
    values = np.empty((n, len(phis)), dtype=float)
    values[0] = rng.normal(scale=scale, size=len(phis))
    innovation_scale = scale * np.sqrt(np.maximum(1.0 - phis**2, 1e-8))
    for index in range(1, n):
        values[index] = phis * values[index - 1] + rng.normal(
            scale=innovation_scale,
            size=len(phis),
        )
    return values


def _average_step_movement(covariances: np.ndarray) -> float:
    values = [
        np.sqrt(max(float(bw2_cov(covariances[index - 1], covariances[index])), 0.0))
        for index in range(1, len(covariances))
    ]
    return float(np.mean(values))


def _step_movements(covariances: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.sqrt(max(float(bw2_cov(covariances[index - 1], covariances[index])), 0.0))
            for index in range(1, len(covariances))
        ],
        dtype=float,
    )


def _standard_error(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def generate_case(
    config: ContinuousCovarianceConfig,
    *,
    replication: int,
    movement_control: float,
    regime: str,
    resolved_seed: int | None = None,
    seed_resample_count: int = 0,
) -> dict[str, object]:
    """Generate one sequence without split-aligned discontinuities."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    requested_seed = config.seed_base + int(replication)
    seed = requested_seed if resolved_seed is None else int(resolved_seed)
    rng = np.random.default_rng(810_311 + seed)
    base_mean = rng.normal(scale=0.25, size=config.d)
    # Hold the starting geometry fixed across rows so the comparison isolates
    # the temporal evolution mechanism rather than a change in covariance
    # spectrum or principal directions.
    base_covariance = _max_commutator_reference(
        config.d,
        condition=config.reference_condition,
        amplitudes=np.asarray(config.cyclic_amplitudes),
        nugget=config.covariance_nugget,
    )
    persistent_direction, local_directions = _directions(base_covariance)
    phase = 2.0 * np.pi * np.arange(config.d) / config.d
    cyclic = np.cos(
        2.0 * np.pi * np.arange(config.n)[:, None] / config.cyclic_period
        + phase[None, :]
    )
    cyclic *= np.asarray(config.cyclic_amplitudes)[None, :]
    cyclic += _stationary_ar(
        rng,
        n=config.n,
        phis=np.full(config.d, -0.35),
        scale=config.cyclic_noise_dispersion,
    )
    fixed_center_cyclic = np.cos(
        2.0
        * np.pi
        * np.arange(config.n)[:, None]
        / config.fixed_center_cyclic_period
        + phase[None, :]
    )
    fixed_center_cyclic *= np.asarray(config.cyclic_amplitudes)[None, :]
    fixed_center_cyclic += _stationary_ar(
        rng,
        n=config.n,
        phis=np.full(config.d, -0.35),
        scale=config.cyclic_noise_dispersion,
    )
    fixed_center_mean_cycle = np.cos(
        2.0
        * np.pi
        * np.arange(config.n)[:, None]
        / config.fixed_center_mean_cycle_period
        + phase[None, :]
    )
    fixed_center_mean_cycle *= np.asarray(config.cyclic_amplitudes)[None, :]
    mean_coordinates = _stationary_ar(
        rng,
        n=config.n,
        phis=np.full(config.d, config.phi),
        scale=config.mean_dispersion,
    )
    means = base_mean + mean_coordinates
    covariances = np.empty((config.n, config.d, config.d), dtype=float)
    reference_means = np.repeat(base_mean[None, :], config.n, axis=0)
    reference_covariances = np.empty_like(covariances)
    minimum_raw = np.inf
    if regime == "persistent":
        multiplier = _calibrate_multiplier(
            target_delta=float(movement_control),
            base_covariance=base_covariance,
            drift_direction=persistent_direction,
        )
        endpoint_deformation = mat_exp(multiplier * persistent_direction)
        endpoint_covariance = endpoint_deformation @ base_covariance @ endpoint_deformation
        endpoint_transport = ot_map(base_covariance, endpoint_covariance)
    for index in range(config.n):
        if regime == "persistent":
            scaled_time = min(index / max(config.val_end, 1), 1.0)
            progress = scaled_time**2 * (3.0 - 2.0 * scaled_time)
            path_transport = (
                (1.0 - progress) * np.eye(config.d)
                + progress * endpoint_transport
            )
            center = path_transport @ base_covariance @ path_transport
            moving_directions = tuple(
                path_transport @ direction @ path_transport
                for direction in local_directions
            )
            moving_directions = tuple(
                _spectral_normalize(direction) for direction in moving_directions
            )
            modes = cyclic[index]
        else:
            center = base_covariance
            moving_directions = local_directions
            modes = float(movement_control) * fixed_center_cyclic[index]
            means[index] = (
                base_mean
                + config.fixed_center_mean_cycle_scale
                * float(movement_control)
                * fixed_center_mean_cycle[index]
            )
        residual = sum(modes[j] * moving_directions[j] for j in range(config.d))
        transport = np.eye(config.d) + residual
        minimum_raw = min(minimum_raw, float(np.linalg.eigvalsh(transport).min()))
        if minimum_raw <= config.minimum_generating_transport_eigenvalue:
            raise ValueError("unsafe latent transport path")
        covariances[index] = transport @ center @ transport
        reference_covariances[index] = center

    observed_steps = _step_movements(covariances)
    reference_steps = _step_movements(reference_covariances)
    fit_boundary_step = float(observed_steps[config.fit_end - 1])
    validation_boundary_step = float(observed_steps[config.val_end - 1])
    reference_fit_boundary_step = float(reference_steps[config.fit_end - 1])
    reference_validation_boundary_step = float(reference_steps[config.val_end - 1])
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(means).tobytes())
    digest.update(np.ascontiguousarray(covariances).tobytes())
    regime_label = (
        "smoothly displaced cyclic covariance process"
        if regime == "persistent"
        else "fixed-center cyclic covariance process"
    )
    return {
        "seed": int(seed),
        "requested_seed": int(requested_seed),
        "seed_resample_count": int(seed_resample_count),
        "means": means,
        "covariances": covariances,
        "reference_means": reference_means,
        "reference_covariances": reference_covariances,
        "sequence_sha256": digest.hexdigest(),
        "metadata": {
            "setting": regime_label,
            "regime": regime,
            "movement_control": float(movement_control),
            "average_step_movement": _average_step_movement(covariances),
            "minimum_step_movement": float(observed_steps.min()),
            "maximum_step_movement": float(observed_steps.max()),
            "fit_boundary_step_movement": fit_boundary_step,
            "validation_boundary_step_movement": validation_boundary_step,
            "fit_boundary_step_to_median_ratio": float(
                fit_boundary_step / max(np.median(observed_steps), 1e-12)
            ),
            "validation_boundary_step_to_median_ratio": float(
                validation_boundary_step / max(np.median(observed_steps), 1e-12)
            ),
            "fit_boundary_reference_step_movement": reference_fit_boundary_step,
            "validation_boundary_reference_step_movement": reference_validation_boundary_step,
            "maximum_reference_step_movement": float(reference_steps.max()),
            "endpoint_reference_distance": float(
                np.sqrt(max(bw2_cov(reference_covariances[0], reference_covariances[-1]), 0.0))
            ),
            "start_reference_distance": 0.0,
            "end_reference_distance": float(
                np.sqrt(max(bw2_cov(reference_covariances[0], reference_covariances[-1]), 0.0))
            ),
            "calibration_multiplier": float(movement_control),
            "generating_transport_clip_count": 0,
            "minimum_raw_generating_transport_eigenvalue": float(minimum_raw),
        },
    }


def resolve_safe_seed(
    config: ContinuousCovarianceConfig,
    replication: int,
) -> tuple[int, int]:
    """Resolve one seed using numerical feasibility only, before evaluation."""

    requested_seed = config.seed_base + int(replication)
    controls = {
        "persistent": config.target_deltas,
        "mean_reverting": config.mean_reversion_controls,
    }
    for attempt in range(config.max_seed_resamples + 1):
        candidate = requested_seed + attempt * config.seed_retry_stride
        try:
            for regime, values in controls.items():
                for movement_control in values:
                    generate_case(
                        config,
                        replication=replication,
                        movement_control=float(movement_control),
                        regime=regime,
                        resolved_seed=candidate,
                        seed_resample_count=attempt,
                    )
        except ValueError as error:
            if str(error) != "unsafe latent transport path":
                raise
            continue
        return int(candidate), int(attempt)
    raise RuntimeError("numerically safe generator seed search exhausted")


def _fixed_coordinates(
    means: np.ndarray,
    covariances: np.ndarray,
    fit_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_mean = means[:fit_end].mean(axis=0)
    reference_covariance, _ = exact_bures_barycenter(covariances[:fit_end])
    coordinates = np.vstack(
        [
            local_bwar_encode(mean, covariance, reference_mean, reference_covariance)
            for mean, covariance in zip(means, covariances)
        ]
    )
    return reference_mean, reference_covariance, coordinates


def _decode_bwar(
    coordinate: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    dimension = len(reference_mean)
    raw_transport = np.eye(dimension) + mat_from_triu(
        np.asarray(coordinate[dimension:]),
        dimension,
    )
    raw_minimum = float(np.linalg.eigvalsh(raw_transport).min())
    mean, covariance = local_bwar_decode(
        coordinate,
        reference_mean,
        reference_covariance,
    )
    return mean, covariance, raw_minimum, raw_minimum <= 0.0


def _forecast_bwar(
    *,
    coordinates: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    ridge: float,
    horizon: int,
    ar_model: str,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    coefficients = fit_var(
        coordinates,
        len(coordinates),
        lam=float(ridge),
        model=ar_model,
    )
    predicted_coordinate = recursive_predict_z(
        coordinates[-1],
        coefficients,
        int(horizon),
    )
    return _decode_bwar(
        predicted_coordinate,
        reference_mean,
        reference_covariance,
    )


def _score_method(
    config: ContinuousCovarianceConfig,
    case: Mapping[str, object],
    *,
    method: str,
    sources: np.ndarray,
    horizon: int,
    ridge: float,
    fixed_reference_mean: np.ndarray,
    fixed_reference_covariance: np.ndarray,
    fixed_coordinates: np.ndarray,
    local_geometry: Mapping[int, object],
    chart_specs: Mapping[str, _MethodSpec],
    chart_coordinates: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    means = np.asarray(case["means"])
    covariances = np.asarray(case["covariances"])
    rows = []
    for source_value in sources:
        source = int(source_value)
        target = source + int(horizon)
        if method == "persistence":
            predicted_mean = means[source]
            predicted_covariance = covariances[source]
            raw_minimum = np.nan
            repaired = False
        elif method == "fixed":
            start = source - config.window_length + 1
            predicted_mean, predicted_covariance, raw_minimum, repaired = (
                _forecast_bwar(
                    coordinates=fixed_coordinates[start : source + 1],
                    reference_mean=fixed_reference_mean,
                    reference_covariance=fixed_reference_covariance,
                    ridge=ridge,
                    horizon=horizon,
                    ar_model=config.ar_model,
                )
            )
        elif method == "local":
            geometry = local_geometry[source]
            predicted_mean, predicted_covariance, raw_minimum, repaired = (
                _forecast_bwar(
                    coordinates=geometry.coordinates,
                    reference_mean=geometry.reference.mean,
                    reference_covariance=geometry.reference.cov,
                    ridge=ridge,
                    horizon=horizon,
                    ar_model=config.ar_model,
                )
            )
        elif method in chart_specs:
            start = source - config.window_length + 1
            coordinates = chart_coordinates[method][start : source + 1]
            coefficients = fit_var(
                coordinates,
                len(coordinates),
                lam=float(ridge),
                model=config.ar_model,
            )
            predicted_coordinate = recursive_predict_z(
                coordinates[-1],
                coefficients,
                int(horizon),
            )
            predicted_mean, predicted_covariance, raw_minimum, repaired = (
                chart_specs[method].decode(predicted_coordinate)
            )
        else:
            raise ValueError(f"unknown method: {method}")
        mean_loss = float(np.sum((predicted_mean - means[target]) ** 2))
        covariance_loss = float(bw2_cov(predicted_covariance, covariances[target]))
        rows.append(
            {
                "source": source,
                "target": target,
                "horizon": int(horizon),
                "method": method,
                "ridge": float(ridge) if method != "persistence" else np.nan,
                "w2_loss": mean_loss + covariance_loss,
                "mean_loss": mean_loss,
                "covariance_loss": covariance_loss,
                "raw_minimum_transport_eigenvalue": raw_minimum,
                "prediction_repaired": bool(repaired),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_one_case(
    config: ContinuousCovarianceConfig,
    *,
    case: Mapping[str, object],
    replication: int,
    target_delta: float,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = np.asarray(case["means"])
    covariances = np.asarray(case["covariances"])
    fixed_mean, fixed_covariance, fixed_coordinates = _fixed_coordinates(
        means,
        covariances,
        config.fit_end,
    )
    chart_specs = {
        spec.name: spec
        for spec in _method_specs(
            d=config.d,
        )
    }
    chart_coordinates = {
        name: np.vstack(
            [spec.encode(mean, covariance) for mean, covariance in zip(means, covariances)]
        )
        for name, spec in chart_specs.items()
    }
    maximum_horizon = max(config.horizons)
    validation_sources = np.arange(
        max(config.window_length - 1, config.fit_end),
        config.val_end - maximum_horizon,
        dtype=int,
    )
    test_sources = np.arange(config.val_end, config.n - maximum_horizon, dtype=int)
    all_sources = np.arange(
        int(validation_sources.min()),
        int(test_sources.max()) + 1,
        dtype=int,
    )
    local_geometry = build_local_bwar_geometry(
        means,
        covariances,
        window_length=config.window_length,
        source_indices=all_sources,
        k_ref=config.k_ref,
        refresh_period=config.refresh_period,
        residual_threshold=config.residual_threshold,
    )
    performance_rows = []
    origin_frames = []
    for horizon in config.horizons:
        selected = {}
        validation_means = {}
        for method in ("fixed", "local", *chart_specs):
            candidates = []
            for ridge in config.ridge_grid:
                validation = _score_method(
                    config,
                    case,
                    method=method,
                    sources=validation_sources,
                    horizon=horizon,
                    ridge=float(ridge),
                    fixed_reference_mean=fixed_mean,
                    fixed_reference_covariance=fixed_covariance,
                    fixed_coordinates=fixed_coordinates,
                    local_geometry=local_geometry,
                    chart_specs=chart_specs,
                    chart_coordinates=chart_coordinates,
                )
                candidates.append((float(validation.w2_loss.mean()), float(ridge)))
            validation_means[method], selected[method] = min(
                candidates,
                key=lambda item: (item[0], -item[1]),
            )

        scored = {}
        for method in METHODS:
            ridge = np.nan if method == "persistence" else selected[method]
            frame = _score_method(
                config,
                case,
                method=method,
                sources=test_sources,
                horizon=horizon,
                ridge=float(ridge),
                fixed_reference_mean=fixed_mean,
                fixed_reference_covariance=fixed_covariance,
                fixed_coordinates=fixed_coordinates,
                local_geometry=local_geometry,
                chart_specs=chart_specs,
                chart_coordinates=chart_coordinates,
            )
            frame.insert(0, "replication", int(replication))
            frame.insert(1, "seed", int(case["seed"]))
            frame.insert(2, "requested_seed", int(case["requested_seed"]))
            frame.insert(3, "seed_resample_count", int(case["seed_resample_count"]))
            frame.insert(4, "regime", regime)
            frame.insert(5, "target_delta", float(target_delta))
            origin_frames.append(frame)
            scored[method] = frame

        fixed_loss = float(scored["fixed"].w2_loss.mean())
        persistence_loss = float(scored["persistence"].w2_loss.mean())
        for method in METHODS:
            frame = scored[method]
            loss = float(frame.w2_loss.mean())
            performance_rows.append(
                {
                    "replication": int(replication),
                    "seed": int(case["seed"]),
                    "requested_seed": int(case["requested_seed"]),
                    "seed_resample_count": int(case["seed_resample_count"]),
                    "regime": regime,
                    "target_delta": float(target_delta),
                    "start_reference_distance": float(case["metadata"]["start_reference_distance"]),
                    "end_reference_distance": float(case["metadata"]["end_reference_distance"]),
                    "horizon": int(horizon),
                    "method": method,
                    "selected_ridge": np.nan if method == "persistence" else float(frame.ridge.iloc[0]),
                    "validation_w2_mean": np.nan if method == "persistence" else float(validation_means[method]),
                    "test_w2_mean": loss,
                    "test_mean_component": float(frame.mean_loss.mean()),
                    "test_covariance_component": float(frame.covariance_loss.mean()),
                    "loss_ratio_to_persistence": loss / max(persistence_loss, 1e-14),
                    "paired_percent_change_vs_fixed": 100.0 * (loss - fixed_loss) / max(fixed_loss, 1e-14),
                    "prediction_repair_count": int(frame.prediction_repaired.sum()),
                    "raw_minimum_transport_eigenvalue": float(frame.raw_minimum_transport_eigenvalue.min(skipna=True)),
                    "n_test_origins": int(len(frame)),
                    "sequence_sha256": str(case["sequence_sha256"]),
                    "generating_transport_clip_count": int(case["metadata"]["generating_transport_clip_count"]),
                    "minimum_raw_generating_transport_eigenvalue": float(case["metadata"]["minimum_raw_generating_transport_eigenvalue"]),
                }
            )

    reference_rows = []
    reference_means = np.asarray(case["reference_means"])
    reference_covariances = np.asarray(case["reference_covariances"])
    for source, geometry in local_geometry.items():
        reference_rows.append(
            {
                "replication": int(replication),
                "seed": int(case["seed"]),
                "regime": regime,
                "target_delta": float(target_delta),
                "origin": int(source),
                "reference_residual": float(geometry.reference.residual),
                "reference_refreshed": bool(geometry.reference.refreshed),
                "reference_fallback": bool(geometry.reference.fallback),
                "reference_w2_squared_to_generating": float(
                    np.sum((geometry.reference.mean - reference_means[int(source)]) ** 2)
                    + bw2_cov(geometry.reference.cov, reference_covariances[int(source)])
                ),
                "minimum_reference_eigenvalue": float(
                    np.linalg.eigvalsh(geometry.reference.cov).min()
                ),
            }
        )
    return (
        pd.DataFrame(performance_rows),
        pd.concat(origin_frames, ignore_index=True),
        pd.DataFrame(reference_rows),
    )


def run_replication(
    replication: int,
    config: ContinuousCovarianceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.validate()
    resolved_seed, seed_resample_count = resolve_safe_seed(config, replication)
    performance_frames = []
    origin_frames = []
    reference_frames = []
    for regime in REGIMES:
        controls = (
            config.target_deltas
            if regime == "persistent"
            else config.mean_reversion_controls
        )
        for movement_control in controls:
            case = generate_case(
                config,
                replication=replication,
                movement_control=float(movement_control),
                regime=regime,
                resolved_seed=resolved_seed,
                seed_resample_count=seed_resample_count,
            )
            performance, origins, references = _evaluate_one_case(
                config,
                case=case,
                replication=replication,
                target_delta=float(movement_control),
                regime=regime,
            )
            performance["movement_control"] = float(movement_control)
            performance["average_step_movement"] = float(case["metadata"]["average_step_movement"])
            for key in (
                "minimum_step_movement",
                "maximum_step_movement",
                "fit_boundary_step_movement",
                "validation_boundary_step_movement",
                "fit_boundary_step_to_median_ratio",
                "validation_boundary_step_to_median_ratio",
                "fit_boundary_reference_step_movement",
                "validation_boundary_reference_step_movement",
                "maximum_reference_step_movement",
                "minimum_raw_generating_transport_eigenvalue",
            ):
                performance[key] = float(case["metadata"][key])
            origins["movement_control"] = float(movement_control)
            references["movement_control"] = float(movement_control)
            performance_frames.append(performance)
            origin_frames.append(origins)
            reference_frames.append(references)
    return (
        pd.concat(performance_frames, ignore_index=True),
        pd.concat(origin_frames, ignore_index=True),
        pd.concat(reference_frames, ignore_index=True),
    )


def summarize_performance(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["regime", "movement_control", "horizon", "method"], as_index=False, sort=False)
        .agg(
            n_replications=("replication", "nunique"),
            average_step_movement=("average_step_movement", "mean"),
            average_step_movement_se=("average_step_movement", _standard_error),
            percent_change_mean=("paired_percent_change_vs_fixed", "mean"),
            percent_change_se=("paired_percent_change_vs_fixed", _standard_error),
            test_w2_mean=("test_w2_mean", "mean"),
            test_w2_se=("test_w2_mean", _standard_error),
            loss_ratio_mean=("loss_ratio_to_persistence", "mean"),
            loss_ratio_se=("loss_ratio_to_persistence", _standard_error),
            repair_rate=("prediction_repair_count", lambda values: float((values > 0).mean())),
        )
    )
    for stem in ("percent_change", "test_w2", "loss_ratio"):
        summary[f"{stem}_ci_low"] = summary[f"{stem}_mean"] - 1.96 * summary[f"{stem}_se"]
        summary[f"{stem}_ci_high"] = summary[f"{stem}_mean"] + 1.96 * summary[f"{stem}_se"]
    return summary


def decision_table(raw: pd.DataFrame) -> pd.DataFrame:
    local = raw.loc[raw["method"].eq("local")].copy()
    table = (
        local.groupby(["regime", "movement_control", "horizon"], as_index=False, sort=False)
        .agg(
            n_replications=("replication", "nunique"),
            average_step_movement=("average_step_movement", "mean"),
            local_minus_fixed_percent=("paired_percent_change_vs_fixed", "mean"),
            percent_se=("paired_percent_change_vs_fixed", _standard_error),
            local_win_rate=("paired_percent_change_vs_fixed", lambda values: float((values < 0.0).mean())),
        )
    )
    table["ci_low"] = table.local_minus_fixed_percent - 1.96 * table.percent_se
    table["ci_high"] = table.local_minus_fixed_percent + 1.96 * table.percent_se
    return table
