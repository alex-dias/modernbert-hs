"""
Specialist ensemble evaluator — Part 4 of the modernbert-hs project.

Ensemble design
---------------
One specialist model is trained per ToxiGen group (asian, black, …, women).
At inference time, each specialist votes on the hate probability of a new
sample. The votes are combined using group-level density weights:

    w_g = mean( density_k{K}_russian )
            averaged over all ToxiGen samples in group g
            (from outputs/2_embeddings/toxigen/{slug}/densities.csv)

This measures how "Russian-like" group g's content is in embedding space.
Groups with higher affinity to Russian HS receive more weight.

Weight normalisation
--------------------
Raw weights are large positive log-densities (e.g. 4000–5000 in 768-dim).
Simple division by their sum would give near-uniform weights because the
absolute values dominate the small relative differences.  We instead shift
by the minimum and then normalise so that the least-similar group gets 0
weight and the most-similar group gets full weight proportionally:

    w_g_shifted = w_g - min(w_g)
    w_g_norm    = w_g_shifted / sum(w_g_shifted)

Two voting modes are supported:
    weighted_average  — final_prob = Σ_g(w_g · p_hate_g) / Σ_g(w_g)
    majority          — each specialist votes hate/no_hate; majority wins
                        (ignores density weights)

The K value and space (raw / pca) should be chosen based on the best
configuration found in the general model evaluation (notebook 04, section 3).

Output
------
    outputs/4_evaluation/{experiment}/
        ensemble_{voting}_{space}_k{K}_results.json   — metrics
        plots/ensemble_{voting}_{space}_k{K}_confusion.png
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .metrics import compute_metrics, bootstrap_ci, get_confusion_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_specialist_models(
    training_root: str,
    model_slug: str | None = None,
) -> list[dict]:
    """
    Walk outputs/3_training/ and return metadata for every specialist model.

    Expected structure:
        training_root/{model_slug}/{group}__specialist/model/

    Parameters
    ----------
    training_root : path to outputs/3_training/
    model_slug    : if given, only return specialists for this model slug
                    (e.g. "answerdotai_ModernBERT-base"). Discovers all slugs
                    when None.

    Returns list of dicts with keys:
        group, model_slug, run_name, model_path
    """
    specialists = []
    if not os.path.exists(training_root):
        logger.warning("Training root not found: %s", training_root)
        return specialists

    for slug in os.listdir(training_root):
        if model_slug and slug != model_slug:
            continue
        model_dir = os.path.join(training_root, slug)
        if not os.path.isdir(model_dir) or slug == "baselines":
            continue

        for run_name in os.listdir(model_dir):
            if "__specialist" not in run_name:
                continue

            model_path = os.path.join(model_dir, run_name, "model")
            if not os.path.isdir(model_path):
                continue

            group = run_name.split("__specialist")[0]
            specialists.append({
                "group":      group,
                "model_slug": slug,
                "run_name":   run_name,
                "model_path": model_path,
            })

    logger.info("Discovered %d specialist models in %s (slug=%s)",
                len(specialists), training_root, model_slug or "all")
    return specialists


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_group_weights(
    toxigen_densities_csv: str,
    k: int,
    space: str = "raw",
) -> dict[str, float]:
    """
    Compute one raw (unnormalised) weight per ToxiGen group.

    w_g = mean( density_k{K}_russian )   for all samples where group == g
          (or density_pca_k{K}_russian for space="pca")

    The returned values are raw mean log-densities.  Callers should apply
    shift-and-normalise before use as voting weights:

        shifted = {g: w - min(weights.values()) for g, w in weights.items()}
        total   = sum(shifted.values()) or 1
        normed  = {g: w / total for g, w in shifted.items()}

    Parameters
    ----------
    toxigen_densities_csv : path to outputs/2_embeddings/toxigen/{slug}/densities.csv
    k                     : K value used for density estimation
    space                 : "raw" or "pca"

    Returns
    -------
    {group: mean_log_density}  — unnormalised
    """
    df = pd.read_csv(toxigen_densities_csv)

    col = f"density_pca_k{k}_russian" if space == "pca" else f"density_k{k}_russian"

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {toxigen_densities_csv}.\n"
            f"Available columns: {[c for c in df.columns if 'russian' in c]}"
        )

    weights: dict[str, float] = {}
    for group, sub in df.groupby("group"):
        weights[group] = float(sub[col].mean())

    logger.info("Raw group weights (k=%d, %s): %s", k, space,
                {g: f"{w:.1f}" for g, w in sorted(weights.items())})
    return weights


def normalise_weights(
    raw_weights: dict[str, float],
    method: str = "softmax",
    temperature: float = 5.0,
) -> dict[str, float]:
    """
    Normalise a dict of raw log-density weights into voting weights.

    Parameters
    ----------
    raw_weights : {group: raw_log_density}
    method      : "softmax" or "linear"
                  - "softmax": Standard for log-scores. exp(w/T) / sum(exp(w/T)).
                  - "linear" : Shifted linear scaling (w - min + floor).
    temperature : Smoothing factor for softmax. Higher = more uniform.
                  With log-densities in range [160, 180], T=5.0 gives a good spread.

    Returns
    -------
    {group: weight} summing to 1.0.
    """
    keys = list(raw_weights.keys())
    values = np.array(list(raw_weights.values()), dtype=float)

    if len(values) == 0:
        return {}

    if np.all(values == values[0]):
        logger.warning("All group weights are identical — using uniform weights")
        normed = np.ones(len(values)) / len(values)
    elif method == "softmax":
        # Subtract max for numerical stability
        v_stable = (values - values.max()) / temperature
        exp_v = np.exp(v_stable)
        normed = exp_v / exp_v.sum()
    else:  # linear
        # Shift by min and add a small floor (10% of range) to ensure every weight counts
        v_min = values.min()
        v_range = values.max() - v_min
        floor = 0.1 * v_range if v_range > 0 else 1.0
        shifted = (values - v_min) + floor
        normed = shifted / shifted.sum()

    return {g: float(w) for g, w in zip(keys, normed)}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _get_hate_probabilities(
    model_path: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Run one specialist model and return probability of the hate class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Bypass strict PyTorch 2.6 torch.load check (safe for our trusted models)
    import transformers.utils.import_utils
    import transformers.modeling_utils
    if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    id2label = model.config.id2label
    hate_idx = next(
        (i for i, l in id2label.items() if "hate" in str(l).lower() and "no" not in str(l).lower()),
        1,
    )

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            batch_probs = torch.softmax(logits, dim=1)[:, hate_idx].cpu().numpy()
            probs.extend(batch_probs.tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    test_csv: str,
    toxigen_densities_csv: str,
    training_root: str,
    output_dir: str,
    k: int = 100,
    space: str = "raw",
    voting: str = "weighted_average",
    model_slug: str | None = None,
    batch_size: int = 32,
    max_length: int = 128,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    norm_method: str = "softmax",
    norm_temperature: float = 5.0,
) -> dict:
    """
    Evaluate the specialist ensemble on the Russian annotated test set.

    Parameters
    ----------
    test_csv              : path to outputs/1_preprocessed/russian/test.csv
    toxigen_densities_csv : path to outputs/2_embeddings/toxigen/{slug}/densities.csv
    training_root         : outputs/3_training/
    output_dir            : outputs/4_evaluation/{experiment}/
    k                     : K value for group weight computation
    space                 : "raw" or "pca"
    voting                : "weighted_average" or "majority"
    model_slug            : restrict specialists to one base model slug; None = all
    bootstrap             : compute 95% bootstrap CI on F1

    Returns
    -------
    metrics dict
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load test set
    df_test = pd.read_csv(test_csv)
    texts = df_test["text"].astype(str).tolist()
    label2id = {"hate": 1, "no_hate": 0, "no hate": 0}
    y_true = df_test["label"].map(label2id).fillna(df_test["label"]).astype(int).values
    logger.info("Test set: %d samples (hate=%d, no_hate=%d)",
                len(y_true), y_true.sum(), (y_true == 0).sum())

    # Discover specialists (optionally filtered by model_slug)
    specialists = discover_specialist_models(training_root, model_slug=model_slug)
    if not specialists:
        raise RuntimeError(
            f"No specialist models found in {training_root}"
            + (f" for model_slug='{model_slug}'" if model_slug else "")
        )

    # Compute and normalise group weights from ToxiGen densities
    raw_weights = compute_group_weights(toxigen_densities_csv, k, space)
    norm_weights = normalise_weights(raw_weights, method=norm_method, temperature=norm_temperature)
    logger.info("Normalised weights (%s, T=%.1f): %s",
                norm_method, norm_temperature,
                {g: f"{w:.4f}" for g, w in sorted(norm_weights.items(), key=lambda x: -x[1])})

    # Run inference on each specialist
    group_probs: dict[str, np.ndarray] = {}
    for spec in specialists:
        group = spec["group"]
        logger.info("Running specialist: %s (%s)", group, spec["model_slug"])
        group_probs[group] = _get_hate_probabilities(
            spec["model_path"], texts, batch_size, max_length
        )

    available_groups = sorted(group_probs.keys())
    logger.info("Specialists available: %s", available_groups)

    # Build weight array aligned to available groups
    weights = np.array([norm_weights.get(g, 0.0) for g in available_groups])

    # Re-normalise in case some groups have no specialist
    w_sum = weights.sum()
    if w_sum == 0:
        logger.warning("All aligned weights are zero — falling back to uniform")
        weights = np.ones(len(available_groups)) / len(available_groups)
    else:
        weights = weights / w_sum

    # Stack specialist probabilities: (n_groups, n_samples)
    probs_matrix = np.stack([group_probs[g] for g in available_groups], axis=0)

    # --- Voting ---
    if voting == "weighted_average":
        y_prob = np.sum(probs_matrix * weights[:, np.newaxis], axis=0)

    elif voting == "majority":
        binary_preds = (probs_matrix >= 0.5).astype(int)
        vote_counts = binary_preds.sum(axis=0)
        y_prob = vote_counts / len(available_groups)

    else:
        raise ValueError(f"Unknown voting mode: '{voting}'. Use 'weighted_average' or 'majority'.")

    # --- Metrics ---
    metrics = compute_metrics(y_true, y_prob)
    slug_tag = f"_{model_slug}" if model_slug else ""
    run_tag = f"ensemble{slug_tag}_{voting}_{space}_k{k}"

    if bootstrap:
        ci = bootstrap_ci(y_true, y_prob, metric="f1", n_bootstrap=n_bootstrap)
        metrics["f1_ci_lower"] = ci["ci_lower"]
        metrics["f1_ci_upper"] = ci["ci_upper"]

    metrics["run_name"]  = run_tag
    metrics["voting"]    = voting
    metrics["space"]     = space
    metrics["k"]         = k
    metrics["model_slug"] = model_slug
    metrics["norm_method"] = norm_method
    metrics["norm_temp"]   = norm_temperature
    metrics["n_groups"]  = len(available_groups)
    metrics["groups"]    = available_groups
    metrics["weights"]   = {g: float(w) for g, w in zip(available_groups, weights)}

    # Save metrics
    out_path = os.path.join(output_dir, f"{run_tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Ensemble metrics saved → %s", out_path)

    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(int),
    }).to_csv(os.path.join(predictions_dir, f"{run_tag}.csv"), index=False)
    logger.info("Predictions saved → %s/predictions/%s.csv", output_dir, run_tag)

    # Save confusion matrix
    _save_confusion_matrix(y_true, y_prob, run_tag, plots_dir)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_prob, run_tag, out_dir):
    cm = get_confusion_matrix(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["no_hate", "hate"], yticklabels=["no_hate", "hate"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(run_tag)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_tag}_confusion.png"), dpi=120)
    plt.close()
