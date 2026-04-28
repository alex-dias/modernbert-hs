"""
Training configuration — shared across all models and datasets.

The goal is not to tune hyperparameters but to compare density-weighted
training across K values, PCA vs raw, and different encoder models.
All experiments use the same config so results are directly comparable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # ---- Model ----
    model_id: str = "answerdotai/ModernBERT-base"
    num_labels: int = 2
    labels: list[str] = field(default_factory=lambda: ["no_hate", "hate"])
    max_length: int = 128

    # ---- Data ----
    # Which density column from Part 2 to use for sample weighting.
    # Pattern: density_k{K}_{group}  |  density_pca_k{K}_{group}  |  density_k{K}_ratio
    # Set to None to train without density weighting.
    density_column: Optional[str] = "density_k5_ratio"

    # ---- Training ----
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 2
    random_state: int = 42
    fp16: bool = False           # set True if GPU supports it
    bf16: bool = False           # set True for Ampere+ GPUs

    # ---- Output ----
    output_root: str = "outputs/3_training"

    # ---- Ollama baselines ----
    ollama_api: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"

    # ---- Unsloth LLM fine-tuning ----
    unsloth_model: str = "unsloth/gemma-4-E2B-it"
    lora_r: int = 8
    lora_alpha: int = 8
    llm_train_size: int = 5000
    llm_max_steps: int = 200
    llm_batch_size: int = 1
    llm_learning_rate: float = 2e-4

    @property
    def label2id(self) -> dict:
        return {l: i for i, l in enumerate(self.labels)}

    @property
    def id2label(self) -> dict:
        return {i: l for i, l in enumerate(self.labels)}

    def model_slug(self) -> str:
        return self.model_id.replace("/", "_")

    def run_name(self, dataset_tag: str) -> str:
        """
        Unique name for one training run, e.g.:
            toxigen__density_k5_ratio
            russian__no_density
        """
        density_tag = self.density_column if self.density_column else "no_density"
        return f"{dataset_tag}__{density_tag}"

    def output_dir(self, dataset_tag: str) -> str:
        return f"{self.output_root}/{self.model_slug()}/{self.run_name(dataset_tag)}"
