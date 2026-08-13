"""Linear-algebra utilities for Gaussian Bures--Wasserstein geometry."""

from __future__ import annotations

import numpy as np


def sym(A):
    return 0.5 * (A + A.T)


def project_spd(A, eps=1e-7):
    A = sym(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def mat_power(A, power, eps=1e-9):
    A = project_spd(A, eps)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None) ** power
    return (vecs * vals) @ vecs.T


def mat_log(A):
    A = project_spd(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.log(np.clip(vals, 1e-12, None))
    return sym((vecs * vals) @ vecs.T)


def mat_exp(A):
    A = sym(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.exp(np.clip(vals, -40, 40))
    return sym((vecs * vals) @ vecs.T)


def bw2_cov(A, B):
    A = project_spd(A)
    B = project_spd(B)
    As = mat_power(A, 0.5)
    inner = As @ B @ As
    val = float(np.trace(A + B - 2.0 * mat_power(inner, 0.5)))
    return max(val, 0.0)


def bw_barycenter(covs, max_iter=35, tol=1e-9):
    S = project_spd(np.mean(covs, axis=0))
    for _ in range(max_iter):
        S_s = mat_power(S, 0.5)
        S_is = mat_power(S, -0.5)
        M = np.zeros_like(S)
        for C in covs:
            M += mat_power(S_s @ C @ S_s, 0.5)
        M /= len(covs)
        S_new = project_spd(S_is @ M @ M @ S_is)
        rel = np.linalg.norm(S_new - S, "fro") / max(np.linalg.norm(S, "fro"), 1e-12)
        S = S_new
        if rel < tol:
            break
    return S


def ot_map(S0, S1):
    S0 = project_spd(S0)
    S1 = project_spd(S1)
    S0_s = mat_power(S0, 0.5)
    S0_is = mat_power(S0, -0.5)
    A = S0_is @ mat_power(S0_s @ S1 @ S0_s, 0.5) @ S0_is
    return sym(A)


def triu_vec(A):
    idx = np.triu_indices(A.shape[0])
    return A[idx]


def mat_from_triu(v, d):
    A = np.zeros((d, d), dtype=float)
    idx = np.triu_indices(d)
    A[idx] = v
    A[(idx[1], idx[0])] = v
    return sym(A)
