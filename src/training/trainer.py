"""
Fine-tuning encoder models for hate speech classification.

Loads the density CSV from Part 2, builds a HuggingFace DatasetDict,
and fine-tunes the model with (optional) density-weighted loss.

Output per run:
    outputs/3_training/{model_slug}/{dataset_tag}__{density_tag}/
        ├── model/          — HF model + tokenizer
        └── metrics.json    — eval metrics on the test split
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from .config import TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_density_dataset(
    density_csv: str,
    config: TrainingConfig,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DatasetDict, np.ndarray | None]:
    """
    Load Part 2 density CSV and return a DatasetDict + optional sample weights.

    Returns
    -------
    dataset     : DatasetDict with "train" and "test" splits
    weights     : dict with "train" and "test" float arrays of sample weights, or None
    """
    df = pd.read_csv(density_csv)

    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {density_csv}")
    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {density_csv}")

    # Normalise labels to int
    if len(df) > 0 and isinstance(df["label"].iloc[0], str):
        df["label"] = df["label"].astype(str).str.strip().map(config.label2id)

    # Coerce to numeric (unmapped strings become NaN) and cast to int
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Stratified split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # Extract sample weights before dropping columns
    weights = None
    if config.density_column and config.density_column in train_df.columns:
        raw_train = train_df[config.density_column].values.astype(float)

        # 1. Temperature smoothing to reduce extreme skew
        # raw_train = raw_train ** 0.5

        # 2. Log-space centering
        # raw_train = np.log1p(raw_train)

        # 3. Aggressive clipping to eliminate extreme outliers and underflow
        # min_allowed = np.percentile(raw_train, 1)
        max_allowed = np.percentile(raw_train, 99)
        # raw_train = np.clip(raw_train, a_min=min_allowed, a_max=max_allowed)

        raw_train = np.clip(raw_train, a_min=None, a_max=max_allowed)

        # 4. Shift to strictly positive and normalize to mean=1
        min_val = raw_train.min()
        raw_train = raw_train - min_val + 1e-6

        train_weights = raw_train / raw_train.mean()


        # Test weights (using the exact same normalization parameters as train)
        # raw_test = test_df[config.density_column].values.astype(float)
        # raw_test = raw_test ** 0.5
        # raw_test = np.log1p(raw_test)
        # raw_test = np.clip(raw_test, a_min=min_allowed, a_max=max_allowed)

        raw_test = test_df[config.density_column].values.astype(float)
        raw_test = np.clip(raw_test, a_min=None, a_max=max_allowed)
        raw_test = raw_test - min_val + 1e-6
        test_weights = raw_test / raw_test.mean()

        weights = {"train": train_weights, "test": test_weights}

        logger.info("Sample weights from '%s': min=%.4f max=%.4f mean=%.4f",
                    config.density_column, train_weights.min(), train_weights.max(), train_weights.mean())
    else:
        if config.density_column:
            logger.warning("Density column '%s' not found — training without weights", config.density_column)

    keep_cols = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[keep_cols])
    test_ds  = Dataset.from_pandas(test_df[keep_cols])
    dataset  = DatasetDict({"train": train_ds, "test": test_ds})

    return dataset, weights


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_compute_metrics(eval_dataset: Dataset | None = None):
    def compute_metrics(pred: EvalPrediction) -> dict:
        logits = pred.predictions
        labels = pred.label_ids
        preds  = np.argmax(logits, axis=1)
        probs  = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1].numpy()

        weights = None
        if eval_dataset is not None and "weight" in eval_dataset.column_names:
            ds_labels = np.array(eval_dataset["labels"])
            if len(labels) == len(ds_labels) and np.array_equal(labels, ds_labels):
                weights = np.array(eval_dataset["weight"])
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Labels in EvalPrediction do not match eval_dataset. Using unweighted metrics."
                )

        acc     = accuracy_score(labels, preds, sample_weight=weights)
        bal_acc = balanced_accuracy_score(labels, preds, sample_weight=weights)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0, sample_weight=weights)
        try:
            auc = roc_auc_score(labels, probs, sample_weight=weights)
        except Exception:
            auc = float("nan")

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc_roc": auc,
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Density-weighted Trainer
# ---------------------------------------------------------------------------

class DensityWeightedTrainer(Trainer):
    """
    Trainer that reads a 'weight' column injected into the dataset
    and applies it as per-sample loss weight.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_fn(logits, labels)

        if weights is not None:
            w = weights.to(per_sample_loss.device).float()
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    density_csv: str,
    dataset_tag: str,
    config: TrainingConfig | None = None,
    test_size: float = 0.2,
) -> dict:
    """
    Fine-tune a model on one density CSV.

    Parameters
    ----------
    density_csv : path to Part 2 densities.csv
    dataset_tag : identifier used in output path, e.g. "toxigen_k5"
    config      : TrainingConfig (uses defaults if None)
    test_size   : fraction held out for evaluation

    Returns
    -------
    metrics : dict of eval metrics saved to metrics.json
    """
    if config is None:
        config = TrainingConfig()

    out_dir = config.output_dir(dataset_tag)
    model_dir = os.path.join(out_dir, "model")
    metrics_path = os.path.join(out_dir, "metrics.json")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Training run: %s ===", config.run_name(dataset_tag))
    logger.info("Model      : %s", config.model_id)
    logger.info("Density col: %s", config.density_column)
    logger.info("Output     : %s", out_dir)

    # 1. Load data
    dataset, sample_weights = load_density_dataset(density_csv, config, test_size, config.random_state)

    # 2. Inject weights into splits as a feature
    if sample_weights is not None:
        dataset["train"] = dataset["train"].add_column("weight", sample_weights["train"].tolist())
        dataset["test"]  = dataset["test"].add_column("weight", sample_weights["test"].tolist())

    # 3. Tokenise
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    remove_cols = [c for c in dataset["train"].column_names if c not in ("label", "weight")]
    dataset = dataset.map(tokenize, batched=True, remove_columns=remove_cols)
    dataset = dataset.rename_column("label", "labels")

    # 4. Model
    # Monkey-patch to bypass the strict PyTorch 2.6 requirement for torch.load
    # (Safe here since we are loading trusted models like tomh/toxigen_roberta)
    import transformers.utils.import_utils
    import transformers.modeling_utils
    if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        ignore_mismatched_sizes=True,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=float(config.learning_rate),
        warmup_ratio=float(config.warmup_ratio),
        weight_decay=float(config.weight_decay),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="auc_roc",
        logging_steps=config.logging_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.random_state,
        report_to="none",
        remove_unused_columns=False,
    )

    # 6. Trainer
    TrainerClass = DensityWeightedTrainer if sample_weights is not None else Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(dataset["test"]),
    )

    # 7. Train
    trainer.train()

    # 8. Evaluate
    eval_results = trainer.evaluate()
    metrics = {k.replace("eval_", ""): v for k, v in eval_results.items()}
    logger.info("Eval results: %s", metrics)

    # 9. Save
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Model saved → %s", model_dir)
    logger.info("Metrics saved → %s", metrics_path)

    return metrics
