"""
Model comparator — Part 4 of the modernbert-hs project.

Discovers all fine-tuned models from outputs/3_training/,
evaluates each on the Russian annotated test set,
and produces a unified comparison DataFrame + plots.

Output:
    outputs/4_evaluation/{experiment_name}/
        results.csv         — one row per model with all metrics
        results_ci.csv      — same with bootstrap CIs
        confusion/          — per-model confusion matrix PNGs
        plots/              — bar charts, heatmaps
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

def discover_models(training_root: str, specialists: bool = False) -> list[dict]:
    """
    Walk outputs/3_training/ and return metadata for every fine-tuned model.

    Expected structure:
        training_root/{model_slug}/{dataset}__{density_tag}/model/

    Returns list of dicts with keys:
        run_name, model_slug, dataset, density_tag, model_path
    """
    models = []
    if not os.path.exists(training_root):
        logger.warning("Training root not found: %s", training_root)
        return models

    for model_slug in os.listdir(training_root):
        model_dir = os.path.join(training_root, model_slug)
        if not os.path.isdir(model_dir) or model_slug == "baselines":
            continue
        for run_name in os.listdir(model_dir):
            is_specialist = "__specialist" in run_name
            if specialists != is_specialist:
                continue

            run_dir = os.path.join(model_dir, run_name)
            model_path = os.path.join(run_dir, "model")
            if not os.path.isdir(model_path):
                continue

            if "__specialist" in run_name:
                group = run_name.split("__specialist")[0]
                dataset = f"spec_{group}"
                density_tag = run_name.replace(f"{group}__", "")
            else:
                parts = run_name.split("__", 1)
                dataset    = parts[0]
                density_tag = parts[1] if len(parts) > 1 else "unknown"

            models.append({
                "run_name":    f"{model_slug}__{run_name}",
                "model_slug":  model_slug,
                "dataset":     dataset,
                "density_tag": density_tag,
                "model_path":  model_path,
            })

    logger.info("Discovered %d fine-tuned models in %s", len(models), training_root)
    return models


def discover_baselines(training_root: str) -> list[dict]:
    """
    Walk outputs/3_training/baselines/ and return LLM baseline metadata.
    """
    baselines = []
    baseline_root = os.path.join(training_root, "baselines")
    if not os.path.exists(baseline_root):
        return baselines

    for model_name in os.listdir(baseline_root):
        for dataset_tag in os.listdir(os.path.join(baseline_root, model_name)):
            results_path = os.path.join(baseline_root, model_name, dataset_tag, "results.json")
            if os.path.exists(results_path):
                baselines.append({
                    "model_name":  model_name,
                    "dataset_tag": dataset_tag,
                    "results_path": results_path,
                })
    return baselines


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_inference_model(model_path: str):
    """
    Load model and tokenizer from path. 
    Returns (model, tokenizer, hate_idx).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Monkey-patch to bypass the strict PyTorch 2.6 requirement for torch.load
    import transformers.utils.import_utils
    import transformers.modeling_utils
    if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Determine which label index corresponds to "hate"
    id2label = model.config.id2label
    hate_idx = next(
        (i for i, l in id2label.items() if "hate" in str(l).lower() and "no" not in str(l).lower()),
        1,
    )
    return model, tokenizer, hate_idx


def _get_probabilities(
    model_path: str = None,
    texts: list[str] = [],
    batch_size: int = 32,
    max_length: int = 128,
    model = None,
    tokenizer = None,
    hate_idx = None,
) -> np.ndarray:
    """Run inference and return probability of the positive (hate) class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    should_unload = False
    if model is None:
        if model_path is None:
            raise ValueError("Either model_path or model/tokenizer/hate_idx must be provided.")
        model, tokenizer, hate_idx = load_inference_model(model_path)
        should_unload = True

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            batch_probs = torch.softmax(logits, dim=1)[:, hate_idx].cpu().numpy()
            probs.extend(batch_probs.tolist())

    if should_unload:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _parse_density_tag(tag: str) -> dict:
    """
    Extract structured fields from density_tag strings like:
        no_density
        density_k5_ratio
        density_pca_k100_ratio
    """
    if "no_density" in tag:
        return {"weighted": False, "space": None, "k": None}

    weighted = True
    space = "pca" if "pca" in tag else "raw"
    k = None
    for part in tag.split("_"):
        if part.startswith("k") and part[1:].isdigit():
            k = int(part[1:])
            break
    return {"weighted": weighted, "space": space, "k": k}


def evaluate_all(
    test_csv: str,
    training_root: str,
    output_dir: str,
    batch_size: int = 32,
    max_length: int = 128,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    base_models: list[str] = None,
    specialists: bool = False,
) -> pd.DataFrame:
    """
    Evaluate all discovered models on the Russian annotated test set.

    Parameters
    ----------
    test_csv      : path to outputs/1_preprocessed/russian/test.csv
    training_root : outputs/3_training/
    output_dir    : outputs/4_evaluation/{experiment}/
    bootstrap     : whether to compute bootstrap CIs

    Returns
    -------
    DataFrame with one row per model and columns for all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    confusion_dir    = os.path.join(output_dir, "confusion")
    plots_dir        = os.path.join(output_dir, "plots")
    predictions_dir  = os.path.join(output_dir, "predictions")
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Load test set
    df_test = pd.read_csv(test_csv)
    texts   = df_test["text"].astype(str).tolist()
    label2id = {"hate": 1, "no_hate": 0, "no hate": 0}
    y_true  = df_test["label"].map(label2id).fillna(df_test["label"]).astype(int).values
    logger.info("Test set: %d samples  (hate=%d, no_hate=%d)", len(y_true), y_true.sum(), (y_true == 0).sum())

    models = discover_models(training_root, specialists=specialists)
    if base_models:
        for b_model in base_models:
            models.append({
                "run_name": f"{b_model.replace('/', '_')}__base_model",
                "model_slug": b_model.replace('/', '_'),
                "dataset": "base_model",
                "density_tag": "no_density",
                "model_path": b_model,
            })
    rows = []

    for meta in models:
        run_name = meta["run_name"]
        logger.info("Evaluating: %s", run_name)

        try:
            y_prob = _get_probabilities(meta["model_path"], texts, batch_size, max_length)
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", run_name, exc)
            continue

        m = compute_metrics(y_true, y_prob)
        tag_info = _parse_density_tag(meta["density_tag"])

        row = {
            "run_name":          run_name,
            "model_slug":        meta["model_slug"],
            "train_dataset":     meta["dataset"],
            "density_tag":       meta["density_tag"],
            "weighted":          tag_info["weighted"],
            "space":             tag_info["space"],
            "k":                 tag_info["k"],
            **m,
        }

        if bootstrap:
            ci = bootstrap_ci(y_true, y_prob, metric="f1", n_bootstrap=n_bootstrap)
            row["f1_ci_lower"] = ci["ci_lower"]
            row["f1_ci_upper"] = ci["ci_upper"]

        rows.append(row)
        _save_confusion_matrix(y_true, y_prob, run_name, confusion_dir)

        safe = run_name.replace("/", "_").replace("\\", "_")
        pd.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": (y_prob >= 0.5).astype(int),
        }).to_csv(os.path.join(predictions_dir, f"{safe}.csv"), index=False)

    # Add LLM baselines (already have metrics, no inference needed)
    for bl in discover_baselines(training_root):
        with open(bl["results_path"]) as f:
            bl_results = json.load(f)
        for mode, m in bl_results.items():
            rows.append({
                "run_name":      f"llm__{bl['model_name']}__{mode}",
                "model_slug":    bl["model_name"],
                "train_dataset": "llm_baseline",
                "density_tag":   mode,
                "weighted":      False,
                "space":         None,
                "k":             None,
                "accuracy":         m.get("accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "precision":     m.get("precision"),
                "recall":        m.get("recall"),
                "f1":            m.get("f1"),
                "auc_roc":       None,
            })

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    logger.info("Results saved → %s/results.csv", output_dir)

    _plot_f1_comparison(results, plots_dir)
    _plot_metric_heatmap(results, plots_dir)

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_prob, run_name, out_dir):
    cm = get_confusion_matrix(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["no_hate", "hate"], yticklabels=["no_hate", "hate"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(run_name[:60])
    plt.tight_layout()
    safe = run_name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=120)
    plt.close()


def _plot_f1_comparison(results: pd.DataFrame, plots_dir: str):
    df = results.dropna(subset=["f1"]).copy()
    df["label"] = df["run_name"].str[:50]

    fig, ax = plt.subplots(figsize=(14, max(5, len(df) * 0.4)))
    colors = ["#e74c3c" if w else "#95a5a6" for w in df["weighted"]]
    bars = ax.barh(df["label"], df["f1"], color=colors)

    if "f1_ci_lower" in df.columns:
        xerr_lower = df["f1"] - df["f1_ci_lower"]
        xerr_upper = df["f1_ci_upper"] - df["f1"]
        ax.errorbar(df["f1"], df["label"],
                    xerr=[xerr_lower, xerr_upper],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xlabel("F1 Score")
    ax.set_title("Model Comparison — F1 on Russian Annotated Test Set")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "f1_comparison.png"), dpi=150)
    plt.close()


def _plot_metric_heatmap(results: pd.DataFrame, plots_dir: str):
    metric_cols = ["accuracy", "balanced_accuracy", "f1", "auc_roc"]
    df = results.dropna(subset=["f1"]).set_index("run_name")[
        [c for c in metric_cols if c in results.columns]
    ].astype(float)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=1, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Metrics — all models on Russian test set")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_heatmap.png"), dpi=150)
    plt.close()
