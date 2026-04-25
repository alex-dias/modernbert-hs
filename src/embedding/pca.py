"""
PCA dimensionality reduction for embeddings.

Each group gets its own PCA fitted on that group's embeddings.
This lets KNN density be computed in each group's own reduced manifold,
so distances between groups are measured in a space tuned to each group.

Two automatic dimension-selection methods are supported:
    elbow     — geometric kneedle on the cumulative-variance curve
    threshold — fewest components reaching a cumulative-variance target
    auto      — max(elbow, threshold)  [default, most conservative]

Fitted PCA objects are saved as .pkl files so downstream steps can reuse them.
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension selection
# ---------------------------------------------------------------------------

def _elbow_dim(evr: np.ndarray) -> int:
    """Geometric kneedle elbow on the cumulative-variance curve."""
    n = len(evr)
    if n <= 1:
        return 1
    cumvar = np.cumsum(evr)
    x = np.linspace(0.0, 1.0, n)
    y = cumvar / cumvar[-1]
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return int(np.argmax(num / den)) + 1


def _threshold_dim(evr: np.ndarray, threshold: float) -> int:
    """Fewest components whose cumulative variance >= threshold."""
    cumvar = np.cumsum(evr)
    indices = np.where(cumvar >= threshold)[0]
    return int(indices[0]) + 1 if len(indices) else len(evr)


def select_n_components(
    evr: np.ndarray,
    method: str = "auto",
    variance_threshold: float = 0.95,
) -> tuple[int, int, int]:
    """
    Returns (n_selected, elbow_dim, threshold_dim).
    method: 'elbow' | 'threshold' | 'auto'
    """
    ed = _elbow_dim(evr)
    td = _threshold_dim(evr, variance_threshold)
    if method == "elbow":
        n = ed
    elif method == "threshold":
        n = td
    else:
        n = min(ed, td)
    return n, ed, td


# ---------------------------------------------------------------------------
# Fit & project
# ---------------------------------------------------------------------------

def fit_pca(
    embeddings: np.ndarray,
    n_components: int | str = "auto",
    variance_threshold: float = 0.95,
    method: str = "auto",
) -> tuple[PCA, int]:
    """
    Fit PCA on *embeddings*.

    Parameters
    ----------
    embeddings        : (N, D) float array — training data for PCA
    n_components      : fixed int, or "auto" to use dimension selection
    variance_threshold: used when n_components=="auto" and method includes threshold
    method            : dimension selection method (auto | elbow | threshold)

    Returns
    -------
    pca   : fitted sklearn PCA object
    n_opt : number of components chosen
    """
    max_comp = min(embeddings.shape[0] - 1, embeddings.shape[1])

    if n_components == "auto":
        # First pass: full PCA to get variance ratios
        pca_full = PCA(n_components=max_comp)
        pca_full.fit(embeddings)
        n_opt, ed, td = select_n_components(pca_full.explained_variance_ratio_, method, variance_threshold)
        logger.info("PCA dim selection — elbow: %d | threshold(%.0f%%): %d | selected: %d",
                    ed, variance_threshold * 100, td, n_opt)
    else:
        n_opt = min(int(n_components), max_comp)
        logger.info("PCA using fixed n_components=%d", n_opt)

    pca = PCA(n_components=n_opt)
    pca.fit(embeddings)
    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    logger.info("PCA fitted: %d components, %.2f%% variance explained", n_opt, cumvar * 100)
    return pca, n_opt


def project(pca: PCA, embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings into PCA space and return float32 array."""
    return pca.transform(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_pca(pca: PCA, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    logger.info("PCA saved → %s", path)


def load_pca(path: str) -> PCA:
    with open(path, "rb") as f:
        pca = pickle.load(f)
    logger.info("PCA loaded ← %s  (%d components)", path, pca.n_components_)
    return pca


# ---------------------------------------------------------------------------
# Per-group PCA
# ---------------------------------------------------------------------------

def fit_group_pcas(
    group_embeddings: dict[str, np.ndarray],
    n_components: int | str = "auto",
    variance_threshold: float = 0.95,
    method: str = "auto",
) -> dict[str, PCA]:
    """
    Fit one PCA per group, each on that group's own embeddings.

    Parameters
    ----------
    group_embeddings : {group_name: (N_group, D) float array}
    n_components     : fixed int or "auto" (applied independently per group)
    variance_threshold / method : passed to fit_pca

    Returns
    -------
    {group_name: fitted PCA}
    """
    group_pcas: dict[str, PCA] = {}
    for grp, emb in group_embeddings.items():
        logger.info("Fitting PCA for group '%s' on shape %s", grp, emb.shape)
        pca, n_opt = fit_pca(emb, n_components, variance_threshold, method)
        logger.info("  → %d components (%.2f%% variance)",
                    n_opt, float(np.cumsum(pca.explained_variance_ratio_)[-1]) * 100)
        group_pcas[grp] = pca
    return group_pcas


def save_group_pcas(group_pcas: dict[str, PCA], out_dir: str) -> None:
    """Save each group's PCA to {out_dir}/{group}.pkl."""
    os.makedirs(out_dir, exist_ok=True)
    for grp, pca in group_pcas.items():
        save_pca(pca, os.path.join(out_dir, f"{grp}.pkl"))


def load_group_pcas(out_dir: str) -> dict[str, PCA]:
    """Load all group PCAs from a directory of {group}.pkl files."""
    group_pcas: dict[str, PCA] = {}
    for fname in os.listdir(out_dir):
        if fname.endswith(".pkl"):
            grp = fname[:-4]
            group_pcas[grp] = load_pca(os.path.join(out_dir, fname))
    return group_pcas


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_variance_curve(pca: PCA, output_path: str, highlight: dict | None = None) -> None:
    """
    Save a cumulative-variance curve.

    Parameters
    ----------
    pca         : fitted PCA (uses all components in explained_variance_ratio_)
    output_path : PNG save path
    highlight   : optional dict {label: n_components} to draw vertical lines
    """
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=1.5)
    ax.axhline(0.95, color="grey", linestyle=":", linewidth=0.8, label="95% variance")

    colors = ["tab:orange", "tab:green", "tab:red"]
    if highlight:
        for (label, n), color in zip(highlight.items(), colors):
            ax.axvline(n, color=color, linestyle="--", label=f"{label} ({n})")

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Variance curve saved → %s", output_path)
