"""
Specialist model training — one fine-tuned model per ToxiGen group.

Each specialist is trained on a single group's data using plain cross-entropy
(no density weighting during training). The density signal is used exclusively
at ensemble voting time.

Output per specialist:
    outputs/3_training/{model_slug}/{group}__specialist/
        ├── model/          — HF model + tokenizer
        └── metrics.json    — eval metrics on the held-out split
"""

import os
import logging
import tempfile
import numpy as np
import pandas as pd

from .config import TrainingConfig
from .trainer import train

logger = logging.getLogger(__name__)

TOXIGEN_GROUPS = [
    "asian", "black", "chinese", "jewish", "latino", "lgbtq",
    "mental_dis", "mexican", "middle_east", "muslim",
    "native_american", "physical_dis", "women",
]


def train_specialist(
    group: str,
    density_csv: str,
    config: TrainingConfig,
) -> dict:
    """
    Train one specialist model on a single ToxiGen group.

    Reads the full ToxiGen densities.csv, filters to the requested group,
    and fine-tunes with plain cross-entropy (density_column=None).

    Parameters
    ----------
    group       : ToxiGen group name, e.g. "asian"
    density_csv : path to outputs/2_embeddings/toxigen/{slug}/densities.csv
    config      : base TrainingConfig — density_column is overridden to None

    Returns
    -------
    metrics dict saved to metrics.json
    """
    df = pd.read_csv(density_csv)

    if "group" not in df.columns:
        raise ValueError(f"'group' column missing in {density_csv}")

    group_df = df[df["group"] == group].reset_index(drop=True)

    if len(group_df) == 0:
        raise ValueError(f"No samples found for group '{group}' in {density_csv}")

    logger.info("[specialist/%s] %d samples", group, len(group_df))

    # Specialist trains with plain CE — no density weighting
    specialist_config = TrainingConfig(
        model_id=config.model_id,
        num_labels=config.num_labels,
        labels=config.labels,
        max_length=config.max_length,
        density_column=None,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        random_state=config.random_state,
        fp16=config.fp16,
        bf16=config.bf16,
        output_root=config.output_root,
    )

    dataset_tag = f"{group}__specialist"

    # Write group-filtered CSV to a temp file for the train() function
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix=f"specialist_{group}_"
    ) as tmp:
        group_df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    try:
        metrics = train(density_csv=tmp_path, dataset_tag=dataset_tag, config=specialist_config)
    finally:
        os.unlink(tmp_path)

    return metrics


def train_all_specialists(
    density_csv: str,
    config: TrainingConfig,
    groups: list[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, dict]:
    """
    Train one specialist per group.

    Parameters
    ----------
    density_csv    : path to toxigen densities.csv
    config         : base TrainingConfig
    groups         : list of groups to train (defaults to all TOXIGEN_GROUPS)
    skip_existing  : if True, skip groups whose metrics.json already exists

    Returns
    -------
    {group: metrics_dict}
    """
    groups = groups or TOXIGEN_GROUPS
    all_metrics: dict[str, dict] = {}

    for group in groups:
        dataset_tag = f"{group}__specialist"
        metrics_path = os.path.join(
            config.output_root,
            config.model_slug(),
            dataset_tag,
            "metrics.json",
        )

        if skip_existing and os.path.exists(metrics_path):
            import json
            logger.info("[specialist/%s] Loading cached metrics", group)
            with open(metrics_path) as f:
                all_metrics[group] = json.load(f)
            continue

        logger.info("=== Training specialist: %s ===", group)
        try:
            all_metrics[group] = train_specialist(group, density_csv, config)
        except Exception as exc:
            logger.error("[specialist/%s] Failed: %s", group, exc)

    return all_metrics
