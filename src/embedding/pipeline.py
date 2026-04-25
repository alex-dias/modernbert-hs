"""
Embedding pipeline — Part 2 of the modernbert-hs project.

Full flow:
    1. Generate embeddings (768-dim) for every dataset with SentenceTransformer.
    2. Split embeddings by group (asian, black, …, russian).
    3. Fit ONE PCA on the combined ToxiGen embeddings (all 13 groups) to find
       the optimal number of dimensions n (auto-selected via elbow / 95% variance).
       Apply that same fitted PCA to ALL groups including Russian → everyone ends
       up in the same shared n-dim space, so cross-group distances are valid.
    4. For each K in K_VALUES compute KNN log-density in two spaces:

       RAW space (768-dim):
           density_k{K}_{g}      — log p(x | reference = group g)
           density_k{K}_all      — log p(x | reference = all groups combined)
           density_k{K}_ratio    — p_russian(x) / p_all(x)

       PCA space (shared n-dim, same transformation for everyone):
           density_pca_k{K}_{g}  — same as raw but in the reduced space
           density_pca_k{K}_all  — same as raw but in the reduced space
           density_pca_k{K}_ratio — p_russian(x) / p_all(x) in PCA space

    5. Save one densities.csv per dataset with all density columns concatenated.

Usage (programmatic):
    from src.embedding.pipeline import run
    run(datasets=["toxigen", "russian"], k_values=[5, 100, 1000])

Usage (CLI):
    python -m src.embedding.pipeline
"""

import os
import logging
import numpy as np
import pandas as pd
import yaml

from .embedder import embed_dataset, embed_texts
from .pca import fit_pca, project, save_pca, load_pca, plot_variance_curve
from .density import knn_log_density, density_ratio
from ..data_preprocessing.russian_processor import RussianProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Step 1 & 2: embed each dataset
# ---------------------------------------------------------------------------

def _embed_all(
    datasets: list[str],
    model_name: str,
    preprocessed_root: str,
    embeddings_root: str,
    batch_size: int,
    use_gpu: bool = False,
) -> dict[str, np.ndarray]:
    """Generate (or load cached) full-split embeddings for every dataset."""
    slug = _model_slug(model_name)
    result = {}

    for ds in datasets:
        out_path = os.path.join(embeddings_root, ds, slug, "full.npy")
        if os.path.exists(out_path):
            logger.info("[%s] Loading cached embeddings from %s", ds, out_path)
            result[ds] = np.load(out_path)
        else:
            result[ds] = embed_dataset(
                dataset_name=ds,
                model_name=model_name,
                preprocessed_root=preprocessed_root,
                output_root=embeddings_root,
                batch_size=batch_size,
                split="full",
                use_gpu=use_gpu,
            )

    return result


# ---------------------------------------------------------------------------
# Step 3: ONE shared PCA fitted on all ToxiGen group embeddings combined
# ---------------------------------------------------------------------------

def _fit_or_load_pca(
    train_datasets: list[str],
    preprocessed_root: str,
    embeddings_map: dict[str, np.ndarray],
    pca_path: str,
    n_components: int | str,
    variance_threshold: float,
    pca_method: str,
) -> object:
    """
    Fit (or load cached) ONE PCA on the combined ToxiGen embeddings.

    The PCA is fitted only on the train_datasets groups (e.g. all ToxiGen
    groups concatenated), then saved to pca_path for reuse.  Russian and any
    other non-training groups are projected with this same PCA so everyone
    lands in the same shared n-dim space.
    """
    if os.path.exists(pca_path):
        logger.info("Loading cached shared PCA from %s", pca_path)
        return load_pca(pca_path)

    # Collect embeddings from train_datasets only
    train_group_embeddings = []
    for ds in train_datasets:
        csv_path = os.path.join(preprocessed_root, ds, "full.csv")
        df = pd.read_csv(csv_path)
        emb = embeddings_map[ds]
        for _, idx in df.groupby("group").groups.items():
            train_group_embeddings.append(emb[list(idx)])

    combined = np.concatenate(train_group_embeddings, axis=0)
    logger.info("Fitting shared PCA on %d ToxiGen samples (shape %s)", len(combined), combined.shape)

    pca, n_opt = fit_pca(combined, n_components, variance_threshold, pca_method)
    save_pca(pca, pca_path)
    return pca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_russian_corpus(
    full_corpus_path: str,
    model_name: str,
    embeddings_root: str,
    batch_size: int,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Embed the full unlabeled Russian corpus (used as KNN reference only).

    Saved to embeddings_root/russian/{slug}/corpus.npy so it is separate
    from the annotated set embeddings (full.npy).
    """
    slug = _model_slug(model_name)
    out_path = os.path.join(embeddings_root, "russian", slug, "corpus.npy")
    if os.path.exists(out_path):
        logger.info("[russian] Loading cached corpus embeddings from %s", out_path)
        return np.load(out_path)

    proc = RussianProcessor(annotated_path="", full_corpus_path=full_corpus_path)
    texts = proc.load_full_corpus()
    logger.info("[russian] Embedding %d texts from full corpus ...", len(texts))
    return embed_texts(texts, model_name, out_path=out_path, batch_size=batch_size, use_gpu=use_gpu)


def _build_group_index(
    datasets: list[str],
    preprocessed_root: str,
    embeddings_map: dict[str, np.ndarray],
    russian_corpus_embeddings: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Split per-dataset raw embeddings into per-group arrays.

    If russian_corpus_embeddings is provided, the 'russian' group is replaced
    with those embeddings (full unlabeled corpus) instead of the smaller
    annotated set, giving KNN density a much larger reference pool.
    """
    group_embeddings: dict[str, np.ndarray] = {}
    for ds in datasets:
        csv_path = os.path.join(preprocessed_root, ds, "full.csv")
        df = pd.read_csv(csv_path)
        emb = embeddings_map[ds]
        for grp, idx in df.groupby("group").groups.items():
            group_embeddings[grp] = emb[list(idx)]

    if russian_corpus_embeddings is not None:
        logger.info("[russian] Using full corpus (%d samples) as KNN reference",
                    len(russian_corpus_embeddings))
        group_embeddings["russian"] = russian_corpus_embeddings

    return group_embeddings, sorted(group_embeddings.keys())


def _raw_density_columns(
    query: np.ndarray,
    group_embeddings: dict[str, np.ndarray],
    all_groups: list[str],
    k: int,
    reference_group: str,
    use_gpu: bool,
) -> dict[str, np.ndarray]:
    """
    Densities in the original embedding space.
    query and all references share the same 768-dim space.
    """
    cols: dict[str, np.ndarray] = {}
    combined_all = np.concatenate([group_embeddings[g] for g in all_groups], axis=0)

    for grp in all_groups:
        cols[f"density_k{k}_{grp}"] = knn_log_density(query, group_embeddings[grp], k, use_gpu=use_gpu)

    cols[f"density_k{k}_all"] = knn_log_density(query, combined_all, k, use_gpu=use_gpu)

    if reference_group in group_embeddings:
        cols[f"density_k{k}_ratio"] = density_ratio(
            cols[f"density_k{k}_{reference_group}"], cols[f"density_k{k}_all"]
        )
    else:
        logger.warning("Reference group '%s' not found — ratio set to NaN", reference_group)
        cols[f"density_k{k}_ratio"] = np.full(len(query), np.nan)

    return cols


def _pca_density_columns(
    query_proj: np.ndarray,
    pca_group_embeddings: dict[str, np.ndarray],
    all_groups: list[str],
    k: int,
    reference_group: str,
    use_gpu: bool,
) -> dict[str, np.ndarray]:
    """
    Densities in the shared PCA space.

    query_proj and all pca_group_embeddings are already projected with the
    SAME PCA, so cross-group distances are valid.
    """
    cols: dict[str, np.ndarray] = {}
    combined_all = np.concatenate([pca_group_embeddings[g] for g in all_groups], axis=0)

    for grp in all_groups:
        cols[f"density_pca_k{k}_{grp}"] = knn_log_density(
            query_proj, pca_group_embeddings[grp], k, use_gpu=use_gpu
        )

    cols[f"density_pca_k{k}_all"] = knn_log_density(query_proj, combined_all, k, use_gpu=use_gpu)

    if reference_group in pca_group_embeddings:
        cols[f"density_pca_k{k}_ratio"] = density_ratio(
            cols[f"density_pca_k{k}_{reference_group}"], cols[f"density_pca_k{k}_all"]
        )
    else:
        logger.warning("Reference group '%s' not found — PCA ratio set to NaN", reference_group)
        cols[f"density_pca_k{k}_ratio"] = np.full(len(query_proj), np.nan)

    return cols


# ---------------------------------------------------------------------------
# Step 4, 5, 6: density (raw + per-group PCA) + save one CSV per dataset
# ---------------------------------------------------------------------------

def _compute_and_save_densities(
    datasets: list[str],
    preprocessed_root: str,
    embeddings_root: str,
    model_name: str,
    raw_embeddings: dict[str, np.ndarray],
    shared_pca: object,
    k_values: list[int],
    use_gpu: bool,
    reference_group: str = "russian",
    russian_corpus_embeddings: np.ndarray | None = None,
) -> None:
    slug = _model_slug(model_name)
    raw_groups, all_groups = _build_group_index(
        datasets, preprocessed_root, raw_embeddings, russian_corpus_embeddings
    )
    logger.info("Groups: %s", all_groups)

    # Project ALL groups with the same shared PCA → valid cross-group distances
    pca_groups = {grp: project(shared_pca, emb) for grp, emb in raw_groups.items()}

    for ds in datasets:
        csv_path = os.path.join(preprocessed_root, ds, "full.csv")
        df = pd.read_csv(csv_path)
        query_raw = raw_embeddings[ds]
        query_pca = project(shared_pca, query_raw)
        out_dir = os.path.join(embeddings_root, ds, slug)
        os.makedirs(out_dir, exist_ok=True)

        row_data: dict = {"text": df["text"], "group": df["group"], "label": df["label"]}

        for k in k_values:
            logger.info("[%s] K=%d — raw embedding space ...", ds, k)
            row_data.update(_raw_density_columns(
                query_raw, raw_groups, all_groups, k, reference_group, use_gpu
            ))

            logger.info("[%s] K=%d — shared PCA space ...", ds, k)
            row_data.update(_pca_density_columns(
                query_pca, pca_groups, all_groups, k, reference_group, use_gpu
            ))

        out_path = os.path.join(out_dir, "densities.csv")
        pd.DataFrame(row_data).to_csv(out_path, index=False)
        logger.info("[%s] Saved → %s  (%d rows × %d cols)", ds, out_path, len(df), len(row_data))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    datasets: list[str] | None = None,
    train_datasets: list[str] | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    k_values: list[int] | None = None,
    preprocessed_root: str = "outputs/1_preprocessed",
    embeddings_root: str = "outputs/2_embeddings",
    n_pca_components: int | str = "auto",
    variance_threshold: float = 0.95,
    pca_method: str = "auto",
    batch_size: int = 32,
    use_gpu: bool = False,
    reference_group: str = "russian",
    config_path: str | None = "configs/datasets.yaml",
) -> None:
    """
    Run the full embedding pipeline.

    train_datasets: datasets whose groups are used to fit the shared PCA
                    (defaults to all non-reference datasets, e.g. ["toxigen"]).
    If config_path is provided, default values are read from it and can be
    overridden by explicitly passing arguments.
    """
    russian_full_corpus_path: str | None = None

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        emb_cfg = cfg.get("embedding", {})
        datasets = datasets or list(cfg.get("datasets", {}).keys())
        train_datasets = train_datasets or cfg.get("train_datasets", [d for d in datasets if d != reference_group])
        k_values = k_values or emb_cfg.get("k_values", [5, 100, 1000])
        model_name = emb_cfg.get("models", [model_name])[0]
        batch_size = emb_cfg.get("batch_size", batch_size)
        # Caller-supplied use_gpu takes precedence; only fall back to config when
        # the function was called with the default (False), meaning the caller
        # didn't explicitly set it.
        if not use_gpu:  # use_gpu is False only when caller didn't override it
            use_gpu = emb_cfg.get("use_gpu", use_gpu)
        n_pca = emb_cfg.get("pca_components", n_pca_components)
        preprocessed_root = cfg.get("preprocessing", {}).get("output_root", preprocessed_root)
        embeddings_root = emb_cfg.get("output_root", embeddings_root)
        russian_full_corpus_path = cfg.get("datasets", {}).get(reference_group, {}).get("full_corpus_path")
    else:
        datasets = datasets or ["toxigen", "russian"]
        train_datasets = train_datasets or [d for d in datasets if d != reference_group]
        k_values = k_values or [5, 100, 1000]
        n_pca = n_pca_components

    slug = _model_slug(model_name)
    pca_path = os.path.join(embeddings_root, "_pca", slug, "shared.pkl")
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)

    logger.info("=== Embedding Pipeline ===")
    logger.info("Datasets      : %s", datasets)
    logger.info("PCA fit on    : %s", train_datasets)
    logger.info("Model         : %s", model_name)
    logger.info("K values      : %s", k_values)
    logger.info("PCA components: %s (%s)", n_pca, pca_method)

    # 1+2: embed all datasets (Russian annotated set → full.npy, used as query)
    embeddings_map = _embed_all(datasets, model_name, preprocessed_root, embeddings_root, batch_size, use_gpu)

    # 2b: embed Russian full corpus for use as KNN reference (separate from annotated set)
    russian_corpus_emb: np.ndarray | None = None
    if reference_group in datasets and russian_full_corpus_path:
        russian_corpus_emb = _embed_russian_corpus(
            russian_full_corpus_path, model_name, embeddings_root, batch_size, use_gpu
        )
    elif reference_group in datasets:
        logger.warning("full_corpus_path not configured — Russian KNN reference uses annotated set only")

    # 3: fit ONE shared PCA on combined train_datasets embeddings
    shared_pca = _fit_or_load_pca(
        train_datasets=train_datasets,
        preprocessed_root=preprocessed_root,
        embeddings_map=embeddings_map,
        pca_path=pca_path,
        n_components=n_pca,
        variance_threshold=variance_threshold,
        pca_method=pca_method,
    )

    # 4+5+6: density (raw 768-dim + shared PCA n-dim) + ratio + save one CSV per dataset
    _compute_and_save_densities(
        datasets, preprocessed_root, embeddings_root, model_name,
        embeddings_map, shared_pca, k_values, use_gpu, reference_group,
        russian_corpus_embeddings=russian_corpus_emb,
    )

    logger.info("=== Embedding Pipeline Complete ===")


if __name__ == "__main__":
    run()
