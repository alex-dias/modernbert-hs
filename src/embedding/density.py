"""
KNN density estimation using FAISS.

Core math: log p(x) ≈ log(k) - log(n) - log(V_d(r_k(x)))
where r_k(x) is the distance to the k-th nearest neighbour in the reference set
and V_d is the volume of a d-dimensional hypersphere.
"""

import numpy as np
import faiss
import logging
from scipy.special import loggamma
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def knn_log_density(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    normalize_vectors: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Estimate log-density of each query point relative to a reference set.

    Parameters
    ----------
    query       : (N, D) float array — points to score
    reference   : (M, D) float array — reference distribution
    k           : number of nearest neighbours
    normalize_vectors : L2-normalise both sets before search (recommended)
    use_gpu     : use FAISS GPU index (falls back to CPU on failure)

    Returns
    -------
    density_log : (N,) float array of log-density estimates
    """
    query = np.array(query, dtype=np.float32)
    reference = np.array(reference, dtype=np.float32)

    if query.ndim == 1:
        query = query.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)

    d = query.shape[1]

    if normalize_vectors:
        query = normalize(query, axis=1)
        reference = normalize(reference, axis=1)

    n = reference.shape[0]
    if k >= n:
        logger.warning("k (%d) >= reference size (%d) — clamping to %d", k, n, n - 1)
        k = max(1, n - 1)

    index = _build_index(reference, d, use_gpu)

    # Search k+1 so we can safely discard self-matches when query == reference
    distances, _ = index.search(query, k + 1)
    kth_distances = distances[:, k]

    # Guard against zero / negative distances (exact duplicates)
    kth_distances = np.maximum(kth_distances, 1e-10)

    log_vol = (d / 2) * np.log(np.pi) + d * np.log(kth_distances) - loggamma(d / 2 + 1)
    density_log = np.log(k) - np.log(n) - log_vol

    return density_log


def density_ratio(density_a: np.ndarray, density_all: np.ndarray) -> np.ndarray:
    """
    Compute the density ratio p_a(x) / p_all(x) in linear scale.
    Inputs are log-densities; output is the exponentiated difference.
    """
    return np.exp(density_a - density_all)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_index(reference: np.ndarray, d: int, use_gpu: bool) -> faiss.Index:
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(reference)
            logger.debug("Using GPU FAISS index")
            return gpu_index
        except Exception as exc:
            logger.warning("GPU FAISS failed (%s) — falling back to CPU", exc)
    index.add(reference)
    return index
