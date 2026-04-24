"""
Embedding generation using SentenceTransformers.

Loads processed CSVs from outputs/1_preprocessed/{dataset}/,
generates embeddings, and saves .npy files to
outputs/2_embeddings/{dataset}/{model_slug}/.
"""

import os
import numpy as np
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def embed_dataset(
    dataset_name: str,
    model_name: str,
    preprocessed_root: str = "outputs/1_preprocessed",
    output_root: str = "outputs/2_embeddings",
    batch_size: int = 32,
    split: str = "full",
) -> np.ndarray:
    """
    Generate and save embeddings for one dataset split.

    Parameters
    ----------
    dataset_name     : e.g. "toxigen" or "russian"
    model_name       : SentenceTransformer model identifier
    preprocessed_root: root of Part 1 outputs
    output_root      : root to save embeddings
    batch_size       : encoding batch size
    split            : "full", "train", or "test"

    Returns
    -------
    embeddings : (N, D) float32 array
    """
    csv_path = os.path.join(preprocessed_root, dataset_name, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Preprocessed file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    logger.info("[%s] Loaded %d texts from %s", dataset_name, len(texts), csv_path)

    model = _load_model(model_name)
    logger.info("[%s] Encoding with %s ...", dataset_name, model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    logger.info("[%s] Embeddings shape: %s", dataset_name, embeddings.shape)

    out_dir = os.path.join(output_root, dataset_name, _model_slug(model_name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.npy")
    np.save(out_path, embeddings)
    logger.info("[%s] Saved embeddings → %s", dataset_name, out_path)

    return embeddings


def embed_texts(
    texts: list[str],
    model_name: str,
    out_path: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed an arbitrary list of texts (e.g. the Russian full corpus).
    Optionally save to out_path.
    """
    model = _load_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, embeddings)
        logger.info("Saved embeddings → %s", out_path)
    return embeddings


# ---------------------------------------------------------------------------
# Model cache — avoid reloading the same model multiple times per run
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        logger.info("Loading model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]
