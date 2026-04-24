"""
LLM zero-shot and few-shot baselines via Ollama.

Runs classification on a test CSV and saves predictions + metrics to
outputs/3_training/baselines/{model_name}/{dataset_tag}/results.json
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_recall_fscore_support,
)

from .config import TrainingConfig

logger = logging.getLogger(__name__)


PROMPT_ZERO_SHOT = """\
Classify the following text as 'hate' or 'no_hate'. Only respond with the label, nothing else.

Text: {text}

Label:"""

PROMPT_WITH_CONTEXT = """\
You are a hate speech classifier. Given a text, output 'hate' if it contains hate speech \
targeting any group, or 'no_hate' if it does not. Only respond with 'hate' or 'no_hate'.

Text: {text}

Classification:"""

FEW_SHOT_EXAMPLES = [
    {"text": "I hate all people from that group, they should not exist.", "label": "hate"},
    {"text": "I disagree with this political decision but everyone deserves respect.", "label": "no_hate"},
    {"text": "Those people are subhuman trash.", "label": "hate"},
    {"text": "Today was a great day at the park with my family.", "label": "no_hate"},
]

PROMPT_FEW_SHOT = """\
Classify each text as 'hate' or 'no_hate'. Only respond with the label.

Examples:
{examples}

Text: {text}
Label:"""


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

class OllamaBaseline:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.api = self.config.ollama_api
        self.model = self.config.ollama_model

    def _call(self, prompt: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0}},
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Ollama call failed after %d retries: %s", max_retries, exc)
                    return None

    @staticmethod
    def _parse(response: str | None) -> str:
        if response is None:
            return "no_hate"
        r = response.lower().strip()
        if "no_hate" in r or "no hate" in r or "not hate" in r:
            return "no_hate"
        if "hate" in r:
            return "hate"
        return "no_hate"

    def _build_prompt(self, text: str, mode: str) -> str:
        if mode == "zero_shot":
            return PROMPT_ZERO_SHOT.format(text=text)
        elif mode == "context":
            return PROMPT_WITH_CONTEXT.format(text=text)
        elif mode == "few_shot":
            examples = "\n".join(
                f"Text: {ex['text']}\nLabel: {ex['label']}" for ex in FEW_SHOT_EXAMPLES
            )
            return PROMPT_FEW_SHOT.format(examples=examples, text=text)
        raise ValueError(f"Unknown mode: {mode}")

    def classify(self, texts: list[str], mode: str = "zero_shot") -> list[str]:
        preds = []
        for i, text in enumerate(texts):
            if (i + 1) % 50 == 0:
                logger.info("[%s] Classified %d/%d", mode, i + 1, len(texts))
            prompt = self._build_prompt(text, mode)
            raw = self._call(prompt)
            preds.append(self._parse(raw))
        return preds


# ---------------------------------------------------------------------------
# Evaluation & saving
# ---------------------------------------------------------------------------

def _compute_metrics(labels: list, preds: list) -> dict:
    label_ids = [1 if l == "hate" else 0 for l in labels]
    pred_ids  = [1 if p == "hate" else 0 for p in preds]

    acc     = accuracy_score(label_ids, pred_ids)
    bal_acc = balanced_accuracy_score(label_ids, pred_ids)
    prec, rec, f1, _ = precision_recall_fscore_support(label_ids, pred_ids, average="binary", zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_samples": len(labels),
    }


def run_baselines(
    test_csv: str,
    dataset_tag: str,
    modes: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, dict]:
    """
    Run all baseline modes on a test CSV and save results.

    Parameters
    ----------
    test_csv    : path to a CSV with columns [text, label]
    dataset_tag : identifier for output path, e.g. "russian_annotated"
    modes       : list of prompt modes to run (default: all three)
    config      : TrainingConfig

    Returns
    -------
    results : {mode: metrics_dict}
    """
    if config is None:
        config = TrainingConfig()
    if modes is None:
        modes = ["zero_shot", "context", "few_shot"]

    df = pd.read_csv(test_csv)
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    out_dir = os.path.join(
        config.output_root, "baselines",
        config.ollama_model.replace(":", "_"), dataset_tag
    )
    os.makedirs(out_dir, exist_ok=True)

    baseline = OllamaBaseline(config)
    all_results: dict[str, dict] = {}

    for mode in modes:
        logger.info("Running baseline mode: %s (%d samples)", mode, len(texts))
        preds = baseline.classify(texts, mode=mode)
        metrics = _compute_metrics(labels, preds)
        all_results[mode] = metrics

        # Save per-sample predictions
        pred_df = df[["text", "label"]].copy()
        pred_df["prediction"] = preds
        pred_df["correct"] = pred_df["label"] == pred_df["prediction"]
        pred_df.to_csv(os.path.join(out_dir, f"{mode}_predictions.csv"), index=False)

        logger.info("[%s] F1=%.4f  Acc=%.4f  BalAcc=%.4f", mode, metrics["f1"], metrics["accuracy"], metrics["balanced_accuracy"])

    # Save combined metrics
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Baseline results saved → %s", results_path)

    return all_results
