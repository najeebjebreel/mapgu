"""MDAV-style k-anonymity clustering and probabilistic permutation-based anonymization."""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from mapgu.utils import get_logger

logger = get_logger(__name__)


def _dist_to_all(x: np.ndarray, X: np.ndarray) -> np.ndarray:
    # x: (d,), X: (n,d)
    # returns (n,)
    return np.linalg.norm(X - x.reshape(1, -1), axis=1)


def _poprow3(X: np.ndarray, y: np.ndarray, idx: np.ndarray, i: int):
    """
    Pop row i from (X, y, idx) and return:
      X2, y2, idx2, pop_x, pop_y, pop_idx
    """
    pop_x = X[i]
    pop_y = y[i]
    pop_idx = idx[i]
    if len(X) <= 1:
        X2 = X[:0]
        y2 = y[:0]
        idx2 = idx[:0]
    else:
        X2 = np.vstack((X[:i], X[i + 1 :]))
        y2 = np.hstack((y[:i], y[i + 1 :]))
        idx2 = np.hstack((idx[:i], idx[i + 1 :]))
    return X2, y2, idx2, pop_x, pop_y, pop_idx


def _cluster_take_k(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pidx: int,
    k: int,
    dist_to_px: Optional[np.ndarray],
):
    """
    Build a cluster of size k by taking k-1 nearest neighbors to px.
    Returns:
      X2, y2, idx2, cluster_idxs (original indices)
    """
    if len(X) == 0:
        return X, y, idx, np.array([pidx], dtype=int)

    distances = _dist_to_all(px, X) if dist_to_px is None else dist_to_px

    take = min(k - 1, len(X))
    nn_local = (
        np.argpartition(distances, take - 1)[:take] if take > 0 else np.array([], dtype=int)
    )

    cluster_idxs = np.hstack(([pidx], idx[nn_local]))
    keep_mask = np.ones(len(X), dtype=bool)
    keep_mask[nn_local] = False

    X2 = X[keep_mask]
    y2 = y[keep_mask]
    idx2 = idx[keep_mask]
    return X2, y2, idx2, cluster_idxs


def _merge_remainder_into_existing_clusters(
    X_orig: np.ndarray,
    clusters: List[np.ndarray],
    rem_idxs: np.ndarray,
):
    """
    Merge remainder points (original indices rem_idxs) into the nearest existing cluster
    based on distance to cluster centroid (computed in ORIGINAL feature space X_orig).
    This guarantees no leftover cluster with size < k.
    """
    if rem_idxs.size == 0:
        return clusters
    if len(clusters) == 0:
        clusters.append(np.array(rem_idxs, dtype=int))
        return clusters

    # Compute centroids of existing clusters in original space
    centroids = np.vstack([X_orig[np.asarray(c, dtype=int)].mean(axis=0) for c in clusters])

    # Assign each remainder point to nearest centroid
    for ridx in np.asarray(rem_idxs, dtype=int):
        d2 = np.sum((centroids - X_orig[ridx]) ** 2, axis=1)
        best = int(np.argmin(d2))
        clusters[best] = np.hstack([clusters[best], ridx])

        # Update centroid of that cluster incrementally (optional).
        # Simplicity > micro-optim: recompute centroid row.
        centroids[best] = X_orig[np.asarray(clusters[best], dtype=int)].mean(axis=0)

    return clusters


def mdav_clusters(X: np.ndarray, y: np.ndarray, k: int, show_progress: bool = True) -> List[np.ndarray]:
    """
    MDAV-style clustering that returns clusters as ORIGINAL row indices.

    Guarantees:
      - Every returned cluster has size >= k (assuming n >= k).
      - The remainder (if < k) is merged into nearest existing clusters.

    Notes:
      - This is an MDAV-like heuristic: pick farthest-from-centroid and farthest-from-that as seeds,
        then take k-1 nearest neighbors for each seed.
      - y is carried along for API compatibility; clustering is done on X only.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    n = int(len(X))
    if n == 0:
        return []
    if n < k:
        raise ValueError(f"Cannot form k-anonymous clusters: n={n} < k={k}")

    X_orig = np.asarray(X)
    Xw = np.array(X_orig, copy=True)
    yw = np.array(y, copy=True)
    idxw = np.arange(len(Xw), dtype=int)

    clusters: List[np.ndarray] = []
    pbar = tqdm(total=len(Xw), disable=not show_progress)

    # Main MDAV loop: form 2 clusters per iteration while enough points exist
    while len(Xw) >= 3 * k:
        xm = Xw.mean(axis=0)

        # pick xr = farthest from centroid
        d_to_cent = _dist_to_all(xm, Xw)
        xri = int(np.argmax(d_to_cent))
        Xw, yw, idxw, xr, yr, idxr = _poprow3(Xw, yw, idxw, xri)

        # pick xs = farthest from xr
        d_to_xr = _dist_to_all(xr, Xw)
        xsi = int(np.argmax(d_to_xr))
        Xw, yw, idxw, xs, ys, idxs = _poprow3(Xw, yw, idxw, xsi)

        # cluster around xr (recompute after popping xs)
        d_to_xr = _dist_to_all(xr, Xw)
        Xw, yw, idxw, c1 = _cluster_take_k(Xw, yw, idxw, xr, yr, idxr, k, d_to_xr)
        clusters.append(np.array(c1, dtype=int))
        pbar.update(len(c1))

        # cluster around xs
        d_to_xs = _dist_to_all(xs, Xw) if len(Xw) > 0 else None
        Xw, yw, idxw, c2 = _cluster_take_k(Xw, yw, idxw, xs, ys, idxs, k, d_to_xs)
        clusters.append(np.array(c2, dtype=int))
        pbar.update(len(c2))

    # Handle remaining points:
    # - if between [2k, 3k): create one k-sized cluster + remainder cluster (size >= k)
    # - else (<2k): remainder might be < k -> merge into existing clusters
    if len(Xw) >= 2 * k and len(Xw) < 3 * k:
        xm = Xw.mean(axis=0)
        d_to_cent = _dist_to_all(xm, Xw)
        xri = int(np.argmax(d_to_cent))
        Xw, yw, idxw, xr, yr, idxr = _poprow3(Xw, yw, idxw, xri)

        d_to_xr = _dist_to_all(xr, Xw) if len(Xw) > 0 else None
        Xw, yw, idxw, c1 = _cluster_take_k(Xw, yw, idxw, xr, yr, idxr, k, d_to_xr)
        clusters.append(np.array(c1, dtype=int))
        pbar.update(len(c1))

        # Remaining is size in [k, 2k) after taking k => remaining >= k always here
        if len(idxw) > 0:
            clusters.append(np.array(idxw, dtype=int))
            pbar.update(len(idxw))
    else:
        # len(Xw) < 2k (or 0). If leftover < k, merge into existing clusters.
        if len(idxw) > 0:
            rem_idxs = np.array(idxw, dtype=int)
            if len(rem_idxs) >= k or len(clusters) == 0:
                clusters.append(rem_idxs)
                pbar.update(len(rem_idxs))
            else:
                clusters = _merge_remainder_into_existing_clusters(X_orig, clusters, rem_idxs)
                pbar.update(len(rem_idxs))

    pbar.close()

    # Final sanity check (guarantee)
    for ci, c in enumerate(clusters):
        if len(c) < k:
            raise RuntimeError(f"Internal error: cluster {ci} has size {len(c)} < k={k}")

    return clusters


def probabilistic_k_anonymize_by_permutation(
    X: np.ndarray,
    y: np.ndarray,
    clusters: List[np.ndarray],
    seed: int = 7,
    protect_mask: Optional[np.ndarray] = None,
    perm_type: str = "rowwise",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probabilistic permutation-based k-anonymization.

    Given clusters of size >= k, randomly permutes the protected (QI) feature
    columns within each cluster. Leaves y unchanged.

    perm_type:
      "rowwise"  (default) — single permutation per cluster: each row receives the
                             entire QI block from another row in the same cluster.
      "colwise"  — independent permutation per QI column within each cluster, so
                   each column is shuffled separately (breaks row-level correlation
                   among QIs but increases the information-theoretic entropy of the
                   anonymized dataset).

    protect_mask: boolean mask of shape (d,), True for columns to permute.
                  If None, ALL feature columns are treated as QIs.
    """
    if perm_type not in ("rowwise", "colwise"):
        raise ValueError(f"perm_type must be 'rowwise' or 'colwise', got {perm_type!r}")

    rng = np.random.default_rng(seed)
    Xk = np.array(X, copy=True)
    yk = np.array(y, copy=True)

    if Xk.ndim != 2:
        raise ValueError("X must be 2D (n, d)")
    if protect_mask is None:
        protect_mask = np.ones(Xk.shape[1], dtype=bool)
    else:
        protect_mask = np.asarray(protect_mask, dtype=bool)
        if protect_mask.shape != (Xk.shape[1],):
            raise ValueError(f"protect_mask must have shape (d,), got {protect_mask.shape} vs d={Xk.shape[1]}")

    qi_cols = np.where(protect_mask)[0]

    for c in clusters:
        c = np.asarray(c, dtype=int)
        if len(c) <= 1:
            continue
        if perm_type == "rowwise":
            # One shared permutation — each row gets an entire QI-block from another row
            perm = rng.permutation(len(c))
            Xk[np.ix_(c, protect_mask)] = Xk[np.ix_(c[perm], protect_mask)]
        else:
            # Independent permutation per QI column
            for col in qi_cols:
                perm = rng.permutation(len(c))
                Xk[c, col] = Xk[c[perm], col]

    return Xk, yk
