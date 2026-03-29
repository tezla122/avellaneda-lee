from __future__ import annotations
import numpy as np
try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    LedoitWolf = None

def sample_covariance(r_centered: np.ndarray, ddof: int=1) -> np.ndarray:
    w, n = r_centered.shape
    denom = max(w - ddof, 1)
    return r_centered.T @ r_centered / denom

def _first_nonzero_positive(v: np.ndarray) -> np.ndarray:
    out = v.astype(float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        for i in range(col.shape[0]):
            if abs(col[i]) > 1e-14:
                if col[i] < 0:
                    out[:, j] *= -1.0
                break
    return out

def _align_with_previous(b_new: np.ndarray, b_prev: np.ndarray | None) -> np.ndarray:
    if b_prev is None or b_prev.shape != b_new.shape:
        return b_new
    out = b_new.copy()
    for k in range(out.shape[1]):
        if np.dot(out[:, k], b_prev[:, k]) < 0:
            out[:, k] *= -1.0
    return out

def pca_loadings_from_cov(cov: np.ndarray, k: int, *, b_prev: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, float]:
    n = cov.shape[0]
    k_eff = min(k, n)
    w, v = np.linalg.eigh(cov)
    w_desc = w[-k_eff:][::-1]
    v_k = v[:, -k_eff:][:, ::-1]
    total = float(np.trace(cov)) if np.trace(cov) > 0 else float(np.sum(w))
    explained = float(np.sum(w_desc)) / total if total > 0 else 0.0
    b = _align_with_previous(v_k, b_prev)
    b = _first_nonzero_positive(b)
    return (b, w_desc, explained)

def maybe_ledoit_wolf_cov(r_centered: np.ndarray, cov: np.ndarray, k: int) -> np.ndarray:
    w, n = r_centered.shape
    rank = np.linalg.matrix_rank(cov, tol=max(1e-09, np.finfo(float).eps * max(cov.shape)))
    need_shrink = n > w - 1 or rank < min(k + 1, n)
    if not need_shrink:
        return cov
    if LedoitWolf is None:
        return cov
    lw = LedoitWolf().fit(r_centered)
    return np.asarray(lw.covariance_, dtype=float)
