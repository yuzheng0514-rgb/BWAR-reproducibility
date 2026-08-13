from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import linalg as scipy_linalg

from bwar.gaussian_geometry import project_spd


_SPD_FLOOR = 1e-8
_SPD_RESOLUTION_ULPS = 64.0
_LOCAL_TRANSPORT_REPAIR_FLOOR = 1e-8
_TRANSPORT_CONGRUENCE_TOL = 1e-7
_EXACT_MAX_ITER = 35
_EXACT_TOL = 1e-9
_ANDERSON_DEPTH = 5
_ACCELERATION_COMPARE_ULPS = 64.0


def _as_float_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    contains_object_complex = array.dtype == object and any(
        isinstance(item, (complex, np.complexfloating)) for item in array.flat
    )
    if np.iscomplexobj(array) or contains_object_complex:
        raise ValueError(f"{name} must be real; complex values are not supported")
    try:
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc


def _as_strict_real_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if np.iscomplexobj(array) or array.dtype.kind not in "iufO":
        raise ValueError(f"{name} must be a real numeric array")
    if array.dtype.kind == "O" and any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in array.flat
    ):
        raise ValueError(f"{name} must be a real numeric array")
    try:
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc


def _immutable_float_array(value: object) -> np.ndarray:
    owned = np.array(value, dtype=float, copy=True, order="C")
    immutable = np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype)
    return immutable.reshape(owned.shape)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


class _BuresNumericalError(FloatingPointError):
    """Numerical failure that makes a Bures covariance update unusable."""


def _validate_positive_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _validated_bures_inputs(
    current: object,
    covs: object,
) -> tuple[np.ndarray, np.ndarray]:
    current_array = _as_float_array(current, name="current")
    covariance_array = _as_float_array(covs, name="covs")
    if current_array.ndim != 2 or current_array.shape[0] != current_array.shape[1]:
        raise ValueError("current must be a square two-dimensional matrix")
    if current_array.shape[0] == 0:
        raise ValueError("Bures matrices must have positive dimension")
    if covariance_array.ndim != 3 or len(covariance_array) == 0:
        raise ValueError("covs must be a nonempty three-dimensional array")
    if covariance_array.shape[1:] != current_array.shape:
        raise ValueError("covs must contain square matrices compatible with current")
    if not np.isfinite(current_array).all() or not np.isfinite(covariance_array).all():
        raise ValueError("current and covs must be finite")
    return current_array, covariance_array


def _is_finite_spd(candidate: object, *, shape: tuple[int, int]) -> bool:
    try:
        array = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    if shape[0] == 0 or array.shape != shape or not np.isfinite(array).all():
        return False
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        return False
    try:
        return bool(np.linalg.eigvalsh(_symmetrize(array)).min() > 0.0)
    except np.linalg.LinAlgError:
        return False


@dataclass(frozen=True)
class _PreparedCovariances:
    projected: np.ndarray
    roots: np.ndarray


def _project_current(current: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = project_spd(current, eps=_SPD_FLOOR)
    except np.linalg.LinAlgError as exc:
        raise _BuresNumericalError("Bures current covariance projection failed") from exc
    if not _is_finite_spd(projected, shape=current.shape):
        raise _BuresNumericalError(
            "Bures current covariance projection produced an invalid SPD matrix"
        )
    return _symmetrize(projected)


def _project_decoded_covariance(matrix: np.ndarray) -> np.ndarray:
    """Project a decoded covariance with a floor resolvable at its scale."""

    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = _symmetrize(project_spd(matrix, eps=_SPD_FLOOR))
        eigenvalues = np.linalg.eigvalsh(projected)
    except np.linalg.LinAlgError as exc:
        raise _BuresNumericalError(
            "decoded covariance projection failed"
        ) from exc
    if not np.isfinite(projected).all() or not np.isfinite(eigenvalues).all():
        raise _BuresNumericalError(
            "decoded covariance projection produced nonfinite values"
        )

    dimension = projected.shape[0]
    scale = max(float(np.abs(eigenvalues).max()), 1.0)
    resolution = (
        _SPD_RESOLUTION_ULPS
        * np.finfo(float).eps
        * dimension
        * scale
    )
    resolved_floor = _SPD_FLOOR + 8.0 * resolution
    if eigenvalues.min() < resolved_floor:
        projected = _symmetrize(
            projected
            + (resolved_floor - float(eigenvalues.min())) * np.eye(dimension)
        )
        try:
            eigenvalues = np.linalg.eigvalsh(projected)
        except np.linalg.LinAlgError as exc:
            raise _BuresNumericalError(
                "decoded covariance repair eigendecomposition failed"
            ) from exc

    if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0.0:
        raise _BuresNumericalError(
            "decoded covariance repair did not produce a finite SPD matrix"
        )
    return projected


def _power_of_projected(matrix: np.ndarray, exponent: float) -> np.ndarray:
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    except np.linalg.LinAlgError as exc:
        raise _BuresNumericalError("Bures projected eigendecomposition failed") from exc
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0.0:
        raise _BuresNumericalError("Bures projected covariance is not finite SPD")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        powered = np.clip(eigenvalues, _SPD_FLOOR, None) ** exponent
        result = (eigenvectors * powered) @ eigenvectors.T
    result = _symmetrize(result)
    if not np.isfinite(result).all():
        raise _BuresNumericalError("Bures projected matrix power became nonfinite")
    return result


def _prepare_covariances(covs: np.ndarray) -> _PreparedCovariances:
    if covs.ndim != 3 or len(covs) == 0 or covs.shape[1] == 0:
        raise ValueError("covs must contain nonempty positive-dimension square matrices")
    if not np.isfinite(covs).all():
        raise ValueError("active window means and covs must be finite")
    projected = []
    roots = []
    for covariance in covs:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                projected_covariance = project_spd(covariance, eps=_SPD_FLOOR)
        except np.linalg.LinAlgError as exc:
            raise _BuresNumericalError(
                "Bures source covariance projection failed"
            ) from exc
        projected_covariance = _symmetrize(projected_covariance)
        if not _is_finite_spd(projected_covariance, shape=covariance.shape):
            raise _BuresNumericalError(
                "Bures source covariance projection produced an invalid SPD matrix"
            )
        projected.append(projected_covariance)
        roots.append(_power_of_projected(projected_covariance, 0.5))
    return _PreparedCovariances(np.array(projected), np.array(roots))


def _robust_svd(factor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return np.linalg.svd(factor, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            return scipy_linalg.svd(
                factor,
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesvd",
            )
        except np.linalg.LinAlgError as fallback_error:
            raise _BuresNumericalError(
                "Bures factor SVD failed with numpy and scipy gesvd drivers"
            ) from fallback_error


def bures_transport_map(reference_cov: object, target_cov: object) -> np.ndarray:
    """Return an SPD Bures map with projected congruence residual at most 1e-7."""

    reference = _as_float_array(reference_cov, name="reference_cov")
    target = _as_float_array(target_cov, name="target_cov")
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1]:
        raise ValueError("reference_cov must be a square two-dimensional matrix")
    if reference.shape[0] == 0:
        raise ValueError("Bures transport covariances must have positive dimension")
    if target.shape != reference.shape:
        raise ValueError("target_cov must have the same square shape as reference_cov")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("reference_cov and target_cov must be finite")

    reference = _project_current(reference)
    target = _project_current(target)
    reference_root = _power_of_projected(reference, 0.5)
    reference_inv_root = _power_of_projected(reference, -0.5)
    target_root = _power_of_projected(target, 0.5)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = target_root @ reference_root
    if not np.isfinite(factor).all():
        raise _BuresNumericalError("Bures transport factor became nonfinite")

    _, singular_values, right_vectors = _robust_svd(factor)
    if not np.isfinite(singular_values).all() or singular_values.min() <= 0.0:
        raise _BuresNumericalError(
            "Bures transport factor SVD produced invalid singular values"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inner_sqrt = (right_vectors.T * singular_values) @ right_vectors
        transport = reference_inv_root @ inner_sqrt @ reference_inv_root
    transport = _symmetrize(transport)
    if not _is_finite_spd(transport, shape=reference.shape):
        raise _BuresNumericalError("Bures transport map is not finite SPD")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reconstructed = transport @ reference @ transport
        congruence_residual = float(
            np.linalg.norm(reconstructed - target, "fro")
            / max(np.linalg.norm(target, "fro"), 1e-12)
        )
    if not np.isfinite(congruence_residual):
        raise _BuresNumericalError(
            "Bures transport congruence residual is nonfinite"
        )
    if congruence_residual > _TRANSPORT_CONGRUENCE_TOL:
        raise _BuresNumericalError(
            "Bures transport congruence residual "
            f"{congruence_residual:.6e} exceeds tolerance "
            f"{_TRANSPORT_CONGRUENCE_TOL:.1e}"
        )
    return transport


def local_bwar_encode(
    mean: object,
    cov: object,
    ref_mean: object,
    ref_cov: object,
) -> np.ndarray:
    """Encode one Gaussian in the Bures chart centered at one reference."""

    mean_array = _as_float_array(mean, name="mean")
    covariance = _as_float_array(cov, name="cov")
    reference_mean = _as_float_array(ref_mean, name="ref_mean")
    reference_covariance = _as_float_array(ref_cov, name="ref_cov")
    if mean_array.ndim != 1 or reference_mean.ndim != 1:
        raise ValueError("mean and ref_mean must be one-dimensional arrays")
    if mean_array.shape[0] == 0:
        raise ValueError("Gaussian chart inputs must have positive dimension")
    if reference_mean.shape != mean_array.shape:
        raise ValueError("mean and ref_mean must have exactly the same shape")
    dimension = mean_array.shape[0]
    expected_covariance_shape = (dimension, dimension)
    if covariance.shape != expected_covariance_shape:
        raise ValueError("cov shape must exactly match the mean dimension")
    if reference_covariance.shape != expected_covariance_shape:
        raise ValueError("ref_cov shape must exactly match the mean dimension")
    if not all(
        np.isfinite(array).all()
        for array in (mean_array, covariance, reference_mean, reference_covariance)
    ):
        raise ValueError("Gaussian chart inputs must be finite")

    transport = bures_transport_map(reference_covariance, covariance)
    covariance_coordinate = (transport - np.eye(dimension))[np.triu_indices(dimension)]
    coordinate = np.concatenate((mean_array - reference_mean, covariance_coordinate))
    if not np.isfinite(coordinate).all():
        raise _BuresNumericalError("local BWAR encoding became nonfinite")
    return coordinate


def local_bwar_decode(
    z: object,
    ref_mean: object,
    ref_cov: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a local chart, repairing only non-SPD maps at the 1e-8 floor."""

    coordinate = _as_strict_real_array(z, name="z")
    reference_mean = _as_strict_real_array(ref_mean, name="ref_mean")
    reference_covariance = _as_strict_real_array(ref_cov, name="ref_cov")
    if coordinate.ndim != 1:
        raise ValueError("z must be a one-dimensional array")
    if reference_mean.ndim != 1 or reference_mean.shape[0] == 0:
        raise ValueError("ref_mean must have positive one-dimensional shape")
    dimension = reference_mean.shape[0]
    if reference_covariance.shape != (dimension, dimension):
        raise ValueError("ref_cov shape must exactly match the reference mean dimension")
    expected_coordinate_dimension = dimension + dimension * (dimension + 1) // 2
    if coordinate.shape != (expected_coordinate_dimension,):
        raise ValueError("z length does not match the reference dimension")
    if not all(
        np.isfinite(array).all()
        for array in (coordinate, reference_mean, reference_covariance)
    ):
        raise ValueError("local BWAR decode inputs must be finite")

    upper_indices = np.triu_indices(dimension)
    transport_increment = np.zeros((dimension, dimension), dtype=float)
    transport_increment[upper_indices] = coordinate[dimension:]
    transport_increment[(upper_indices[1], upper_indices[0])] = coordinate[dimension:]
    transport = np.eye(dimension) + transport_increment
    try:
        transport_eigenvalues = np.linalg.eigvalsh(transport)
    except np.linalg.LinAlgError as exc:
        raise _BuresNumericalError(
            "local BWAR transport eigendecomposition failed"
        ) from exc
    if not np.isfinite(transport_eigenvalues).all():
        raise _BuresNumericalError("local BWAR transport eigenvalues are nonfinite")
    if transport_eigenvalues.min() <= 0.0:
        try:
            transport = _symmetrize(
                project_spd(transport, eps=_LOCAL_TRANSPORT_REPAIR_FLOOR)
            )
        except np.linalg.LinAlgError as exc:
            raise _BuresNumericalError(
                "local BWAR non-SPD transport repair failed"
            ) from exc
        if not _is_finite_spd(transport, shape=(dimension, dimension)):
            raise _BuresNumericalError(
                "local BWAR non-SPD transport repair produced an invalid map"
            )

    projected_reference = _project_current(reference_covariance)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        decoded_mean = reference_mean + coordinate[:dimension]
        raw_covariance = transport @ projected_reference @ transport
    if not np.isfinite(decoded_mean).all() or not np.isfinite(raw_covariance).all():
        raise _BuresNumericalError("local BWAR decode output became nonfinite")
    decoded_covariance = _project_decoded_covariance(raw_covariance)
    return decoded_mean, decoded_covariance


def _factor_svd_step(
    current: np.ndarray,
    prepared: _PreparedCovariances,
) -> np.ndarray:
    shape = prepared.projected.shape[1:]
    if not _is_finite_spd(current, shape=shape):
        raise _BuresNumericalError("Bures fixed-point current covariance is not finite SPD")
    current_root = _power_of_projected(current, 0.5)
    current_inv_root = _power_of_projected(current, -0.5)
    average = np.zeros_like(current)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for covariance_root in prepared.roots:
                factor = covariance_root @ current_root
                _, singular_values, right_vectors = _robust_svd(factor)
                if not np.isfinite(singular_values).all():
                    raise _BuresNumericalError(
                        "Bures factor SVD produced nonfinite singular values"
                    )
                square_root = (
                    right_vectors.T * singular_values
                ) @ right_vectors
                average += _symmetrize(square_root)
            average /= len(prepared.roots)
            factor = average @ current_inv_root
            raw_candidate = _symmetrize(factor.T @ factor)
    except np.linalg.LinAlgError as exc:
        raise _BuresNumericalError("Bures factor SVD failed") from exc

    if not np.isfinite(raw_candidate).all():
        raise _BuresNumericalError(
            "Bures fixed-point step produced a nonfinite raw covariance"
        )
    if not _is_finite_spd(raw_candidate, shape=shape):
        raise _BuresNumericalError(
            "Bures fixed-point step produced a raw covariance that is not SPD"
        )
    candidate = project_spd(raw_candidate, eps=_SPD_FLOOR)
    candidate = _symmetrize(candidate)
    if not _is_finite_spd(candidate, shape=shape):
        raise _BuresNumericalError(
            "Bures fixed-point final projection produced an invalid SPD covariance"
        )
    return candidate


def _prepared_residual(
    current: np.ndarray,
    prepared: _PreparedCovariances,
) -> tuple[float, np.ndarray]:
    step = _factor_svd_step(current, prepared)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = np.linalg.norm(step - current, "fro")
        denominator = max(np.linalg.norm(current, "fro"), 1e-12)
        residual = float(numerator / denominator)
    return residual, step


def _select_safeguarded_candidate(
    ordinary_candidate: np.ndarray,
    accelerated_candidate: np.ndarray,
    prepared: _PreparedCovariances,
) -> tuple[np.ndarray, float, bool]:
    ordinary_residual, _ = _prepared_residual(ordinary_candidate, prepared)
    if not np.isfinite(ordinary_residual):
        raise _BuresNumericalError(
            "exact Bures ordinary candidate residual became nonfinite"
        )
    try:
        accelerated_candidate = _project_current(
            _symmetrize(accelerated_candidate)
        )
        accelerated_residual, _ = _prepared_residual(
            accelerated_candidate,
            prepared,
        )
    except _BuresNumericalError:
        return ordinary_candidate, ordinary_residual, False

    comparison_tolerance = (
        _ACCELERATION_COMPARE_ULPS
        * np.finfo(float).eps
        * max(1.0, abs(ordinary_residual))
    )
    if (
        np.isfinite(accelerated_residual)
        and accelerated_residual <= ordinary_residual + comparison_tolerance
    ):
        return accelerated_candidate, accelerated_residual, True
    return ordinary_candidate, ordinary_residual, False


@dataclass(frozen=True)
class ReferenceState:
    """One immutable rolling reference and its trailing-window metadata."""

    origin: int
    mean: np.ndarray
    cov: np.ndarray
    window_start: int
    window_stop: int
    residual: float
    refreshed: bool
    fallback: bool

    def __post_init__(self) -> None:
        for name in ("mean", "cov"):
            object.__setattr__(self, name, _immutable_float_array(getattr(self, name)))


@dataclass(frozen=True)
class LocalGeometry:
    """One immutable local chart and all trailing-window coordinates in it."""

    origin: int
    reference: ReferenceState
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ReferenceState):
            raise ValueError("reference must be a ReferenceState")
        origin = _validate_positive_integer(self.origin, name="origin", minimum=0)
        reference_origin = _validate_positive_integer(
            self.reference.origin,
            name="reference.origin",
            minimum=0,
        )
        if origin != reference_origin:
            raise ValueError("origin must equal reference.origin")
        window_start = _validate_positive_integer(
            self.reference.window_start,
            name="reference.window_start",
            minimum=0,
        )
        window_stop = _validate_positive_integer(
            self.reference.window_stop,
            name="reference.window_stop",
            minimum=0,
        )
        if window_stop <= window_start:
            raise ValueError("reference window must have positive length")

        reference_mean = _as_float_array(self.reference.mean, name="reference.mean")
        reference_covariance = _as_float_array(
            self.reference.cov,
            name="reference.cov",
        )
        if reference_mean.ndim != 1 or reference_mean.shape[0] == 0:
            raise ValueError("reference mean must have positive one-dimensional shape")
        dimension = reference_mean.shape[0]
        if reference_covariance.shape != (dimension, dimension):
            raise ValueError("reference mean and covariance dimensions are incompatible")
        if not np.isfinite(reference_mean).all() or not np.isfinite(
            reference_covariance
        ).all():
            raise ValueError("reference mean and covariance must be finite")

        coordinates = _as_strict_real_array(self.coordinates, name="coordinates")
        if coordinates.ndim != 2:
            raise ValueError("coordinates must be a two-dimensional array")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        expected_rows = window_stop - window_start
        expected_columns = dimension + dimension * (dimension + 1) // 2
        if coordinates.shape != (expected_rows, expected_columns):
            raise ValueError(
                "coordinates shape must match the reference window and dimension"
            )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "coordinates", _immutable_float_array(coordinates))


def bures_fixed_point_step(current: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Apply one Bures-Wasserstein covariance barycenter fixed-point step."""

    current_array, covariance_array = _validated_bures_inputs(current, covs)
    current_spd = _project_current(current_array)
    prepared = _prepare_covariances(covariance_array)
    return _factor_svd_step(current_spd, prepared)


def fixed_point_residual(current: np.ndarray, covs: np.ndarray) -> float:
    """Return the relative Frobenius residual of one fixed-point step."""

    current_array, covariance_array = _validated_bures_inputs(current, covs)
    current_spd = _project_current(current_array)
    prepared = _prepare_covariances(covariance_array)
    residual, _ = _prepared_residual(current_spd, prepared)
    return residual


def exact_bures_barycenter(
    covs: object,
    *,
    max_iter: int = _EXACT_MAX_ITER,
    tolerance: float = _EXACT_TOL,
) -> tuple[np.ndarray, float]:
    """Return the safeguarded exact covariance barycenter and final residual."""

    covariance_array = _as_float_array(covs, name="covs")
    if covariance_array.ndim != 3:
        raise ValueError("covs must be a three-dimensional array")
    if (
        len(covariance_array) == 0
        or covariance_array.shape[1] == 0
        or covariance_array.shape[1] != covariance_array.shape[2]
    ):
        raise ValueError("covs must contain nonempty positive-dimension square matrices")
    max_iter_value = _validate_positive_integer(
        max_iter,
        name="max_iter",
        minimum=1,
    )
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a finite positive real scalar")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be a finite positive real scalar")
    prepared = _prepare_covariances(covariance_array)
    covariance, residual = _solve_exact_barycenter(
        prepared,
        max_iter=max_iter_value,
        tolerance=tolerance_value,
    )
    return np.array(covariance, copy=True), float(residual)


def _solve_exact_barycenter(
    prepared: _PreparedCovariances,
    *,
    max_iter: int = _EXACT_MAX_ITER,
    tolerance: float = _EXACT_TOL,
) -> tuple[np.ndarray, float]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        initial = np.mean(prepared.projected, axis=0)
    if not np.isfinite(initial).all():
        raise _BuresNumericalError(
            "exact Bures solver arithmetic initialization became nonfinite"
        )
    current = _project_current(initial)
    last_residual = np.inf
    residual_history: list[np.ndarray] = []
    step_history: list[np.ndarray] = []
    ordinary_recovery: np.ndarray | None = None
    for iteration in range(max_iter):
        recovery_for_current = ordinary_recovery
        try:
            residual, step = _prepared_residual(current, prepared)
        except _BuresNumericalError:
            if recovery_for_current is None:
                raise
            current = recovery_for_current
            ordinary_recovery = None
            recovery_for_current = None
            residual, step = _prepared_residual(current, prepared)
        if not np.isfinite(residual):
            if recovery_for_current is None:
                raise _BuresNumericalError(
                    "exact Bures solver residual became nonfinite"
                )
            current = recovery_for_current
            ordinary_recovery = None
            recovery_for_current = None
            residual, step = _prepared_residual(current, prepared)
            if not np.isfinite(residual):
                raise _BuresNumericalError(
                    "exact Bures ordinary recovery residual became nonfinite"
                )
        last_residual = residual
        if residual <= tolerance:
            try:
                checked_residual, _ = _prepared_residual(current, prepared)
            except _BuresNumericalError:
                if recovery_for_current is None:
                    raise
                current = recovery_for_current
                ordinary_recovery = None
                continue
            if not np.isfinite(checked_residual):
                if recovery_for_current is not None:
                    current = recovery_for_current
                    ordinary_recovery = None
                    continue
                raise _BuresNumericalError(
                    "exact Bures solver final residual became nonfinite"
                )
            if checked_residual <= tolerance:
                return current, checked_residual
            last_residual = checked_residual
        if iteration == max_iter - 1:
            break
        ordinary_recovery = None

        fixed_point_difference = step - current
        residual_history.append(fixed_point_difference)
        step_history.append(step)
        depth = min(_ANDERSON_DEPTH, len(residual_history) - 1)
        candidate = step
        accepted_acceleration = False
        if depth > 0:
            recent_residuals = residual_history[-(depth + 1) :]
            recent_steps = step_history[-(depth + 1) :]
            residual_differences = np.column_stack(
                [
                    (recent_residuals[index + 1] - recent_residuals[index]).ravel()
                    for index in range(depth)
                ]
            )
            step_differences = np.column_stack(
                [
                    (recent_steps[index + 1] - recent_steps[index]).ravel()
                    for index in range(depth)
                ]
            )
            try:
                coefficients = np.linalg.lstsq(
                    residual_differences,
                    fixed_point_difference.ravel(),
                    rcond=None,
                )[0]
            except np.linalg.LinAlgError:
                coefficients = None
            if coefficients is not None:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    accelerated = step - (
                        step_differences @ coefficients
                    ).reshape(step.shape)
                if np.isfinite(accelerated).all():
                    candidate, _, accepted_acceleration = (
                        _select_safeguarded_candidate(
                            step,
                            accelerated,
                            prepared,
                        )
                    )
        current = candidate
        if accepted_acceleration:
            ordinary_recovery = step
    raise _BuresNumericalError(
        "exact Bures solver did not converge within "
        f"{max_iter} iterations; final residual {last_residual:.6e}"
    )


class RollingBuresReference:
    """Track a trailing-window Bures reference with bounded warm updates."""

    def __init__(
        self,
        window_length: int,
        k_ref: int = 3,
        refresh_period: int = 24,
        residual_threshold: float = 1e-4,
    ) -> None:
        self.window_length = _validate_positive_integer(
            window_length,
            name="window_length",
            minimum=3,
        )
        self.k_ref = _validate_positive_integer(k_ref, name="k_ref", minimum=1)
        self.refresh_period = _validate_positive_integer(
            refresh_period,
            name="refresh_period",
            minimum=1,
        )
        if isinstance(residual_threshold, (bool, np.bool_)) or not isinstance(
            residual_threshold,
            Real,
        ):
            raise ValueError("residual_threshold must be a finite nonnegative real scalar")
        try:
            threshold = float(residual_threshold)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "residual_threshold must be a finite nonnegative real scalar"
            ) from exc
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("residual_threshold must be a finite nonnegative real scalar")
        self.residual_threshold = threshold
        self._state: ReferenceState | None = None
        self._window_means: np.ndarray | None = None
        self._window_covs: np.ndarray | None = None

    @staticmethod
    def _validated_series(
        means: object,
        covs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_array = _as_float_array(means, name="means")
        covariance_array = _as_float_array(covs, name="covs")
        if mean_array.ndim != 2:
            raise ValueError("means must be a two-dimensional array")
        if covariance_array.ndim != 3:
            raise ValueError("covs must be a three-dimensional array")
        if covariance_array.shape[1] != covariance_array.shape[2]:
            raise ValueError("covs must contain square matrices")
        if mean_array.shape[1] == 0 or covariance_array.shape[1] == 0:
            raise ValueError("means and covs must have positive dimension")
        if len(mean_array) != len(covariance_array):
            raise ValueError("means and covs must have the same number of observations")
        if mean_array.shape[1] != covariance_array.shape[1]:
            raise ValueError("means and covs must have compatible dimensions")
        return mean_array, covariance_array

    @staticmethod
    def _exact_covariance(
        prepared: _PreparedCovariances,
    ) -> tuple[np.ndarray, float]:
        return _solve_exact_barycenter(
            prepared,
            max_iter=_EXACT_MAX_ITER,
            tolerance=_EXACT_TOL,
        )

    def reference_at(
        self,
        origin: int,
        means: np.ndarray,
        covs: np.ndarray,
    ) -> ReferenceState:
        """Return the reference for observations in ``[origin - L + 1, origin + 1)``."""

        origin_value = _validate_positive_integer(origin, name="origin", minimum=0)
        if self._state is not None and origin_value != self._state.origin + 1:
            raise ValueError("rolling reference updates must be sequential")

        mean_array, covariance_array = self._validated_series(means, covs)
        if origin_value < self.window_length - 1 or origin_value >= len(mean_array):
            raise ValueError("origin does not provide a complete reference window")
        window_start = origin_value - self.window_length + 1
        window_stop = origin_value + 1
        window_means = mean_array[window_start:window_stop]
        window_covs = covariance_array[window_start:window_stop]

        if self._state is not None:
            if (
                self._state.mean.shape != mean_array.shape[1:]
                or self._state.cov.shape != covariance_array.shape[1:]
            ):
                raise ValueError("means and covs are incompatible with the prior reference")
            previous_means = mean_array[
                self._state.window_start : self._state.window_stop
            ]
            previous_covs = covariance_array[
                self._state.window_start : self._state.window_stop
            ]
            if not np.array_equal(previous_means, self._window_means) or not np.array_equal(
                previous_covs,
                self._window_covs,
            ):
                raise ValueError(
                    "sequential inputs must preserve append-only history"
                )

        if not np.isfinite(window_means).all() or not np.isfinite(window_covs).all():
            raise ValueError("active window means and covs must be finite")
        prepared = _prepare_covariances(window_covs)

        if self._state is None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                reference_mean = window_means.mean(axis=0)
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                reference_mean = self._state.mean + (
                    window_means[-1]
                    - self._window_means[0]
                ) / self.window_length
        if not np.isfinite(reference_mean).all():
            raise ValueError("rolling reference mean became nonfinite")

        initial = self._state is None
        refreshed = initial or origin_value % self.refresh_period == 0
        fallback = False
        if refreshed:
            final_covariance, residual = self._exact_covariance(prepared)
        else:
            candidate = self._state.cov
            warm_invalid = False
            for _ in range(self.k_ref):
                try:
                    candidate = _factor_svd_step(candidate, prepared)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
                    warm_invalid = True
                    break
                if not _is_finite_spd(candidate, shape=window_covs.shape[1:]):
                    warm_invalid = True
                    break

            if not warm_invalid:
                try:
                    warm_residual, _ = _prepared_residual(
                        candidate,
                        prepared,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
                    warm_invalid = True
                else:
                    warm_invalid = (
                        not np.isfinite(warm_residual)
                        or warm_residual > self.residual_threshold
                    )
                    if not warm_invalid:
                        final_covariance = candidate
                        residual = warm_residual
            if warm_invalid:
                final_covariance, residual = self._exact_covariance(prepared)
                fallback = True
        if not np.isfinite(residual):
            raise _BuresNumericalError("final Bures reference residual is nonfinite")

        state = ReferenceState(
            origin=origin_value,
            mean=reference_mean,
            cov=final_covariance,
            window_start=window_start,
            window_stop=window_stop,
            residual=residual,
            refreshed=refreshed,
            fallback=fallback,
        )
        self._state = state
        self._window_means = _immutable_float_array(window_means)
        self._window_covs = _immutable_float_array(window_covs)
        return state


@dataclass(frozen=True)
class DualLaggedRidgeAR1:
    """Memory-efficient ridge lag-design AR(1) fit.

    The fitted coefficient matrix is represented as ``projection @ response``
    so a high-dimensional coordinate system does not require storing or
    inverting a full coordinate-by-coordinate matrix.
    """

    coordinate_mean: np.ndarray
    projection: np.ndarray
    response: np.ndarray

    def __post_init__(self) -> None:
        for name in ("coordinate_mean", "projection", "response"):
            object.__setattr__(self, name, _immutable_float_array(getattr(self, name)))

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict the next coordinate row from one finite fitted-shape state."""

        state = _as_float_array(state, name="state")
        if state.shape != self.coordinate_mean.shape:
            raise ValueError("state shape does not match fitted AR state shape")
        if not np.isfinite(state).all():
            raise ValueError("state must be finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prediction = (
                self.coordinate_mean
                + ((state - self.coordinate_mean) @ self.projection) @ self.response
            )
        if not np.isfinite(prediction).all():
            raise ValueError("prediction became nonfinite for the supplied state")
        return prediction

    def predict_recursive(self, state: np.ndarray, horizon: int) -> np.ndarray:
        """Feed each one-step prediction back as state for ``horizon`` steps."""

        if (
            isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError("horizon must be an integer at least one")
        predicted = state
        for _ in range(horizon):
            predicted = self.predict(predicted)
        return predicted


def fit_dual_lagged_ridge_ar1(
    coordinates: np.ndarray,
    *,
    ridge: float,
) -> DualLaggedRidgeAR1:
    """Fit the article's ridge lag-design AR(1) in dual form.

    This is algebraically identical to the full ``p=1`` estimator in
    :func:`bwar.paper_jcgs.gaussian_models.fit_var`, but solves an
    observation-by-observation system when the coordinate dimension is larger
    than the local fitting window.
    """

    coordinates = _as_float_array(coordinates, name="coordinates")
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if len(coordinates) < 3:
        raise ValueError("at least three coordinate rows are required")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if isinstance(ridge, (bool, np.bool_)) or not isinstance(ridge, Real):
        raise ValueError("ridge must be a finite real scalar greater than zero")
    try:
        ridge_value = float(ridge)
    except (ValueError, OverflowError) as exc:
        raise ValueError("ridge must be a finite real scalar greater than zero") from exc
    if not np.isfinite(ridge_value) or ridge_value <= 0:
        raise ValueError("ridge must be a finite real scalar greater than zero")

    n = len(coordinates)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinate_mean = coordinates.mean(axis=0)
    if not np.isfinite(coordinate_mean).all():
        raise ValueError(
            "nonfinite coordinate mean while fitting ridge lag-design AR(1)"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = coordinates - coordinate_mean
    if not np.isfinite(centered).all():
        raise ValueError(
            "nonfinite centered coordinates while fitting ridge lag-design AR(1)"
        )

    lagged = centered[:-1]
    response = centered[1:]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = lagged @ lagged.T
    if not np.isfinite(gram).all():
        raise ValueError(
            "nonfinite Gram matrix while fitting ridge lag-design AR(1)"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        system = gram + (n - 1) * ridge_value * np.eye(n - 1)
    if not np.isfinite(system).all():
        raise ValueError(
            "nonfinite regularized Gram matrix while fitting ridge "
            "lag-design AR(1)"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            response_coef = np.linalg.solve(system, response)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "dual ridge lag-design system could not be solved"
            ) from exc
        projection = lagged.T
    if not np.isfinite(projection).all():
        raise ValueError("nonfinite dual ridge lag-design projection")
    if not np.isfinite(response_coef).all():
        raise ValueError("nonfinite dual ridge lag-design response coefficients")
    return DualLaggedRidgeAR1(coordinate_mean, projection, response_coef)


@dataclass(frozen=True)
class PrimalRidgeAR1:
    """Primal ridge AR(1) fit with owned, read-only parameter arrays."""

    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x_mean", "y_mean", "coef"):
            object.__setattr__(self, name, _immutable_float_array(getattr(self, name)))

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict the next row from one finite fitted-shape state."""

        state = _as_float_array(state, name="state")
        if state.shape != self.x_mean.shape:
            raise ValueError("state shape does not match fitted AR state shape")
        if not np.isfinite(state).all():
            raise ValueError("state must be finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prediction = self.y_mean + (state - self.x_mean) @ self.coef
        if not np.isfinite(prediction).all():
            raise ValueError("prediction became nonfinite for the supplied state")
        return prediction

    def predict_recursive(self, state: np.ndarray, horizon: int) -> np.ndarray:
        """Feed each one-step prediction back as state for ``horizon`` steps."""

        if (
            isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError("horizon must be an integer at least one")
        predicted = state
        for _ in range(horizon):
            predicted = self.predict(predicted)
        return predicted


def fit_primal_ridge_ar1(coordinates: np.ndarray, *, ridge: float) -> PrimalRidgeAR1:
    """Fit the ridge lag-design AR(1) normal equations in primal form."""

    coordinates = _as_float_array(coordinates, name="coordinates")
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if len(coordinates) < 3:
        raise ValueError("at least three coordinate rows are required")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if isinstance(ridge, (bool, np.bool_)) or not isinstance(ridge, Real):
        raise ValueError("ridge must be a finite real scalar greater than zero")
    try:
        ridge_value = float(ridge)
    except (ValueError, OverflowError) as exc:
        raise ValueError("ridge must be a finite real scalar greater than zero") from exc
    if not np.isfinite(ridge_value) or ridge_value <= 0:
        raise ValueError("ridge must be a finite real scalar greater than zero")

    coordinate_mean = coordinates.mean(axis=0)
    X = coordinates[:-1] - coordinate_mean
    Y = coordinates[1:] - coordinate_mean
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_mean = coordinate_mean
        y_mean = coordinate_mean
    if not np.isfinite(x_mean).all() or not np.isfinite(y_mean).all():
        raise ValueError("nonfinite intermediate means while fitting primal ridge AR(1)")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_centered = X
        y_centered = Y
    if not np.isfinite(x_centered).all() or not np.isfinite(y_centered).all():
        raise ValueError("nonfinite centered coordinates while fitting primal ridge AR(1)")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = x_centered.T @ x_centered
        cross_product = x_centered.T @ y_centered
    if not np.isfinite(gram).all() or not np.isfinite(cross_product).all():
        raise ValueError("nonfinite normal equations while fitting primal ridge AR(1)")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        system = gram + len(X) * ridge_value * np.eye(coordinates.shape[1])
    if not np.isfinite(system).all():
        raise ValueError("nonfinite regularized Gram matrix while fitting primal ridge AR(1)")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            coef = np.linalg.solve(system, cross_product)
        except np.linalg.LinAlgError as exc:
            raise ValueError("primal ridge system could not be solved") from exc
    if not np.isfinite(coef).all():
        raise ValueError("nonfinite primal coefficients while fitting primal ridge AR(1)")
    return PrimalRidgeAR1(x_mean, y_mean, coef)


def common_source_indices(
    n_states: int,
    horizon: int,
    max_window_length: int = 48,
) -> np.ndarray:
    """Return source-state indices shared by every compared online method."""

    n_states = _validate_positive_integer(n_states, name="n_states", minimum=1)
    horizon = _validate_positive_integer(horizon, name="horizon", minimum=1)
    max_window_length = _validate_positive_integer(
        max_window_length,
        name="max_window_length",
        minimum=1,
    )
    first_source = max_window_length - 1
    stop = n_states - horizon
    if stop <= first_source:
        raise ValueError("no valid source remains after burn-in and forecast horizon")
    return np.arange(first_source, stop, dtype=int)


def forecast_online_encoded(
    coordinates: object,
    source_index: int,
    window_length: int,
    ridge: float,
    horizon: int,
    decode: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and recursively forecast one causal trailing encoded-state window."""

    coordinates = _as_strict_real_array(coordinates, name="coordinates")
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if coordinates.shape[1] == 0:
        raise ValueError("coordinates must have positive dimension")
    source_index = _validate_positive_integer(
        source_index,
        name="source_index",
        minimum=0,
    )
    window_length = _validate_positive_integer(
        window_length,
        name="window_length",
        minimum=3,
    )
    horizon = _validate_positive_integer(horizon, name="horizon", minimum=1)
    if not callable(decode):
        raise ValueError("decode must be callable")
    if source_index >= len(coordinates):
        raise ValueError("source_index is outside coordinates")
    window_start = source_index - window_length + 1
    if window_start < 0:
        raise ValueError("source_index does not provide the requested trailing window")
    active_coordinates = coordinates[window_start : source_index + 1]
    if not np.isfinite(active_coordinates).all():
        raise ValueError("active coordinates must be finite")

    model = fit_dual_lagged_ridge_ar1(
        active_coordinates,
        ridge=ridge,
    )
    predicted_coordinate = model.predict_recursive(
        coordinates[source_index],
        horizon=horizon,
    )
    decoded = decode(predicted_coordinate)
    if not isinstance(decoded, (tuple, list)) or len(decoded) != 2:
        raise ValueError("decode must return a mean and covariance pair")
    decoded_mean = _as_strict_real_array(decoded[0], name="decoded mean")
    decoded_covariance = _as_strict_real_array(
        decoded[1],
        name="decoded covariance",
    )
    if decoded_mean.ndim != 1 or decoded_mean.shape[0] == 0:
        raise ValueError("decoded mean must have positive one-dimensional shape")
    dimension = decoded_mean.shape[0]
    if decoded_covariance.shape != (dimension, dimension):
        raise ValueError("decoded covariance shape is incompatible with decoded mean")
    if not np.isfinite(decoded_mean).all():
        raise ValueError("decoded mean must be finite")
    if not np.isfinite(decoded_covariance).all():
        raise ValueError("decoded covariance must be finite")
    try:
        final_covariance = _project_decoded_covariance(decoded_covariance)
    except _BuresNumericalError as exc:
        raise ValueError("decoded covariance could not be projected to SPD") from exc
    return decoded_mean, final_covariance


def forecast_online_raw_var_window(
    raw: object,
    source_state_index: int,
    state_window_size: int,
    window_length: int,
    ridge: float,
    horizon: int,
) -> np.ndarray:
    """Forecast the raw rows comprising target Gaussian state ``t + h``."""

    raw = _as_strict_real_array(raw, name="raw")
    if raw.ndim != 2:
        raise ValueError("raw must be a two-dimensional array")
    if raw.shape[1] == 0:
        raise ValueError("raw must have positive dimension")
    source_state_index = _validate_positive_integer(
        source_state_index,
        name="source_state_index",
        minimum=0,
    )
    state_window_size = _validate_positive_integer(
        state_window_size,
        name="state_window_size",
        minimum=1,
    )
    window_length = _validate_positive_integer(
        window_length,
        name="window_length",
        minimum=1,
    )
    horizon = _validate_positive_integer(horizon, name="horizon", minimum=1)

    source_stop = (source_state_index + 1) * state_window_size
    if source_stop > len(raw):
        raise ValueError("source_state_index extends beyond available raw rows")
    lookback = window_length * state_window_size
    training_start = source_stop - lookback
    if training_start < 0:
        raise ValueError("source state does not provide the requested raw lookback")
    training = raw[training_start:source_stop]
    if not np.isfinite(training).all():
        raise ValueError("training raw rows must be finite")
    model = fit_primal_ridge_ar1(training, ridge=ridge)

    total_steps = horizon * state_window_size
    target_window = np.empty((state_window_size, raw.shape[1]), dtype=float)
    predicted = training[-1]
    target_start = total_steps - state_window_size
    for step in range(total_steps):
        predicted = model.predict(predicted)
        if step >= target_start:
            target_window[step - target_start] = predicted
    return target_window


def forecast_online_raw_var_mean(
    raw: object,
    source_state_index: int,
    state_window_size: int,
    window_length: int,
    ridge: float,
    horizon: int,
) -> np.ndarray:
    """Return the column mean of the forecast raw target-state window."""

    target_window = forecast_online_raw_var_window(
        raw,
        source_state_index,
        state_window_size,
        window_length,
        ridge,
        horizon,
    )
    return target_window.mean(axis=0)


def _validated_source_indices(source_indices: object) -> tuple[int, ...]:
    try:
        sources = np.asarray(source_indices, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("source_indices must be a finite 1D integer sequence") from exc
    if sources.ndim != 1:
        raise ValueError("source_indices must be a finite 1D integer sequence")
    if len(sources) == 0:
        return ()
    if any(
        isinstance(source, (bool, np.bool_))
        or not isinstance(source, (int, np.integer))
        for source in sources
    ):
        raise ValueError("source_indices must be a finite 1D integer sequence")
    values = tuple(int(source) for source in sources)
    if any(right - left != 1 for left, right in zip(values, values[1:])):
        raise ValueError("source_indices must be consecutive without duplicates or gaps")
    return values


def build_local_bwar_geometry(
    means: object,
    covs: object,
    window_length: int,
    source_indices: object,
    k_ref: int = 3,
    refresh_period: int = 24,
    residual_threshold: float = 1e-4,
) -> dict[int, LocalGeometry]:
    """Build causal trailing-window BWAR charts at consecutive source origins."""

    sources = _validated_source_indices(source_indices)
    if not sources:
        return {}

    tracker = RollingBuresReference(
        window_length=window_length,
        k_ref=k_ref,
        refresh_period=refresh_period,
        residual_threshold=residual_threshold,
    )
    mean_array, covariance_array = tracker._validated_series(means, covs)
    if sources[0] < tracker.window_length - 1 or sources[-1] >= len(mean_array):
        raise ValueError("source_indices contain an out-of-range origin")

    geometries: dict[int, LocalGeometry] = {}
    for origin in sources:
        reference = tracker.reference_at(origin, mean_array, covariance_array)
        coordinates = np.vstack(
            [
                local_bwar_encode(
                    mean_array[index],
                    covariance_array[index],
                    reference.mean,
                    reference.cov,
                )
                for index in range(reference.window_start, reference.window_stop)
            ]
        )
        geometries[origin] = LocalGeometry(origin, reference, coordinates)
    return geometries


def forecast_local_bwar(
    geometry: LocalGeometry,
    ridge: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recursively forecast and decode within one local BWAR reference chart."""

    if not isinstance(geometry, LocalGeometry):
        raise ValueError("geometry must be a LocalGeometry")
    reference_mean = _as_float_array(geometry.reference.mean, name="reference mean")
    reference_covariance = _as_float_array(
        geometry.reference.cov,
        name="reference covariance",
    )
    if reference_mean.ndim != 1 or reference_mean.shape[0] == 0:
        raise ValueError("geometry reference mean must have positive one-dimensional shape")
    dimension = reference_mean.shape[0]
    if reference_covariance.shape != (dimension, dimension):
        raise ValueError("geometry reference covariance shape is incompatible with its mean")
    if not np.isfinite(reference_mean).all() or not np.isfinite(reference_covariance).all():
        raise ValueError("geometry reference must be finite")
    expected_coordinate_dimension = dimension + dimension * (dimension + 1) // 2
    if geometry.coordinates.shape[1:] != (expected_coordinate_dimension,):
        raise ValueError("geometry coordinate dimension is incompatible with its reference")

    model = fit_dual_lagged_ridge_ar1(
        geometry.coordinates,
        ridge=ridge,
    )
    predicted_coordinate = model.predict_recursive(
        geometry.coordinates[-1],
        horizon=horizon,
    )
    return local_bwar_decode(
        predicted_coordinate,
        geometry.reference.mean,
        geometry.reference.cov,
    )
