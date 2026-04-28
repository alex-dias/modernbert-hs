"""
LLM-based hate speech classification.

Two modes:
  1. Zero-shot / few-shot baselines via Ollama (no training required)
  2. Instruction fine-tuning via Unsloth + LoRA (requires GPU + unsloth)

Ollama outputs:
    outputs/3_training/baselines/{model_name}/{dataset_tag}/
        {mode}_predictions.csv
        results.json

Unsloth outputs:
    outputs/3_training/llm_finetuned/{model_slug}/{dataset_tag}/
        model/               — saved LoRA adapter
        metrics.json         — eval metrics on the test split
        predictions.csv      — per-sample predictions
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


# ---------------------------------------------------------------------------
# Prompt templates  (Ollama zero-shot / few-shot)
# ---------------------------------------------------------------------------

# Simple binary — minimal instruction, works well for models that follow
# strict output constraints.
PROMPT_ZERO_SHOT = """\
You are a hate speech detection system. Read the text below and output \
exactly one label: 'hate' if the text contains hate speech targeting any \
person or group, or 'no_hate' if it does not. Do not explain your answer.

Text: {text}

Label:"""

# Role-play framing — positions the model as an expert annotator, which
# tends to improve calibration on borderline cases.
PROMPT_ROLEPLAY = """\
You are an expert hate speech annotator working on a content moderation \
research project. Your task is to classify Russian-origin social media \
texts that may contain hate speech. Apply strict annotation guidelines: \
label as 'hate' only if the text explicitly dehumanises, threatens, or \
calls for discrimination against a person or group based on ethnicity, \
religion, gender, sexual orientation, or disability. Otherwise label \
as 'no_hate'. Output only the label, nothing else.

Text: {text}

Label:"""

# Few-shot — concrete examples covering clear hate, borderline political
# criticism (should be no_hate), and neutral content.
_FEW_SHOT_EXAMPLES = [
    {
        "text": "All people from that ethnic group are criminals and should be expelled.",
        "label": "hate",
        "note": "explicit dehumanisation + call for discrimination",
    },
    {
        "text": "I strongly disagree with the government's immigration policy.",
        "label": "no_hate",
        "note": "political criticism, no group dehumanisation",
    },
    {
        "text": "Those people are subhuman trash and don't deserve to live here.",
        "label": "hate",
        "note": "clear dehumanisation",
    },
    {
        "text": "The match last night was incredible, our team played brilliantly.",
        "label": "no_hate",
        "note": "neutral",
    },
    {
        "text": "Members of that religion are all terrorists and should be banned.",
        "label": "hate",
        "note": "religious group targeted with generalisation",
    },
    {
        "text": "I think we need stricter border controls to manage migration flows.",
        "label": "no_hate",
        "note": "policy opinion without targeting a group",
    },
]

_FEW_SHOT_BLOCK = "\n\n".join(
    f"Text: {ex['text']}\nLabel: {ex['label']}"
    for ex in _FEW_SHOT_EXAMPLES
)

PROMPT_FEW_SHOT = """\
You are an expert hate speech annotator. Classify each text as 'hate' or \
'no_hate'. Output only the label — 'hate' if the text explicitly \
dehumanises, threatens, or incites discrimination against any person or \
group; 'no_hate' otherwise.

Examples:
{examples}

Text: {{text}}
Label:""".format(examples=_FEW_SHOT_BLOCK)

# Map mode name → template string
PROMPT_TEMPLATES: dict[str, str] = {
    "zero_shot": PROMPT_ZERO_SHOT,
    "roleplay":  PROMPT_ROLEPLAY,
    "few_shot":  PROMPT_FEW_SHOT,
}


# ---------------------------------------------------------------------------
# Ollama baseline
# ---------------------------------------------------------------------------

class OllamaBaseline:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.api    = self.config.ollama_api
        self.model  = self.config.ollama_model

    def _call(self, prompt: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
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

    def classify(self, texts: list[str], mode: str = "zero_shot") -> list[str]:
        if mode not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(PROMPT_TEMPLATES)}")
        template = PROMPT_TEMPLATES[mode]
        preds = []
        for i, text in enumerate(texts):
            if (i + 1) % 50 == 0:
                logger.info("[%s] Classified %d/%d", mode, i + 1, len(texts))
            prompt = template.format(text=text)
            raw    = self._call(prompt)
            preds.append(self._parse(raw))
        return preds


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(labels: list, preds: list) -> dict:
    label_ids = [1 if str(l) == "hate" else 0 for l in labels]
    pred_ids  = [1 if p == "hate" else 0 for p in preds]

    acc     = accuracy_score(label_ids, pred_ids)
    bal_acc = balanced_accuracy_score(label_ids, pred_ids)
    prec, rec, f1, _ = precision_recall_fscore_support(
        label_ids, pred_ids, average="binary", zero_division=0
    )
    return {
        "accuracy":          float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision":         float(prec),
        "recall":            float(rec),
        "f1":                float(f1),
        "n_samples":         len(labels),
    }


# ---------------------------------------------------------------------------
# Baseline runner  (public API — same signature as old baselines.py)
# ---------------------------------------------------------------------------

def run_baselines(
    test_csv: str,
    dataset_tag: str,
    modes: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, dict]:
    """
    Run Ollama zero-shot / roleplay / few-shot on a test CSV.

    Parameters
    ----------
    test_csv    : CSV with columns [text, label]
    dataset_tag : identifier for output path, e.g. "russian_annotated"
    modes       : prompt modes to run (default: all three)
    config      : TrainingConfig

    Returns
    -------
    {mode: metrics_dict}
    """
    if config is None:
        config = TrainingConfig()
    if modes is None:
        modes = list(PROMPT_TEMPLATES.keys())

    df     = pd.read_csv(test_csv)
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    out_dir = os.path.join(
        config.output_root, "baselines",
        config.ollama_model.replace(":", "_"), dataset_tag,
    )
    os.makedirs(out_dir, exist_ok=True)

    baseline     = OllamaBaseline(config)
    all_results: dict[str, dict] = {}

    for mode in modes:
        logger.info("Running baseline mode: %s (%d samples)", mode, len(texts))
        preds   = baseline.classify(texts, mode=mode)
        metrics = _compute_metrics(labels, preds)
        all_results[mode] = metrics

        pred_df = df[["text", "label"]].copy()
        pred_df["prediction"] = preds
        pred_df["correct"]    = pred_df["label"] == pred_df["prediction"]
        pred_df.to_csv(os.path.join(out_dir, f"{mode}_predictions.csv"), index=False)

        logger.info(
            "[%s] F1=%.4f  Acc=%.4f  BalAcc=%.4f",
            mode, metrics["f1"], metrics["accuracy"], metrics["balanced_accuracy"],
        )

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Baseline results saved → %s", results_path)

    return all_results


# ---------------------------------------------------------------------------
# Unsloth fine-tuning
# ---------------------------------------------------------------------------

# Instruction template for SFT — kept identical to PROMPT_ZERO_SHOT so the
# fine-tuned model learns to answer the same prompt used at inference time.
_FT_CLASSIFY_INSTRUCTION = (
    "You are a hate speech detection system. Read the text below and output "
    "exactly one label: 'hate' if the text contains hate speech targeting any "
    "person or group, or 'no_hate' if it does not. Do not explain your answer."
)


def _build_conversations(df: pd.DataFrame) -> list[list[dict]]:
    """Convert a DataFrame with [text, label] to a list of chat conversations."""
    convs = []
    for _, row in df.iterrows():
        lbl = "hate" if str(row["label"]) in ("hate", "1", "1.0") else "no_hate"
        user_msg = f"{_FT_CLASSIFY_INSTRUCTION}\n\nText: {row['text']}\n\nLabel:"
        convs.append([
            {"role": "user",      "content": [{"type": "text", "text": user_msg}]},
            {"role": "assistant", "content": [{"type": "text", "text": lbl}]},
        ])
    return convs


def _parse_ft_response(response: str | None) -> str:
    """Same parser used by OllamaBaseline, reused for the fine-tuned model."""
    return OllamaBaseline._parse(response)


def train_llm(
    density_csv: str,
    test_csv: str,
    dataset_tag: str,
    k: int,
    space: str = "raw",
    config: TrainingConfig | None = None,
) -> dict:
    """
    Fine-tune an instruction LLM with Unsloth LoRA on density-weighted data.

    Parameters
    ----------
    density_csv : outputs/2_embeddings/{dataset}/{model_slug}/densities.csv
    test_csv    : CSV with [text, label] for final evaluation
    dataset_tag : identifier, e.g. "toxigen"
    k           : K value used to select the density column
    space       : 'raw' or 'pca'
    config      : TrainingConfig

    Returns
    -------
    metrics dict (also saved to outputs/3_training/llm_finetuned/.../metrics.json)
    """
    # Lazy imports so the rest of the module works without GPU dependencies.
    try:
        import torch
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from datasets import Dataset as HFDataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise ImportError(
            "Unsloth fine-tuning requires: unsloth, trl, datasets. "
            f"Missing: {e}"
        )

    if config is None:
        config = TrainingConfig()

    # ---- select density column ----
    col_prefix = "density_pca" if space == "pca" else "density"
    density_col = f"{col_prefix}_k{k}_ratio"

    df_train = pd.read_csv(density_csv)
    if density_col not in df_train.columns:
        available = [c for c in df_train.columns if c.startswith("density")]
        raise ValueError(
            f"Column '{density_col}' not found in {density_csv}.\n"
            f"Available density columns: {available}"
        )

    # ---- density-weighted sampling ----
    weights = df_train[density_col].values.copy()
    weights = np.clip(weights, 0, None)
    total   = weights.sum()
    if total == 0:
        raise ValueError(f"All weights are zero for column '{density_col}'.")
    weights = weights / total

    train_size = config.llm_train_size
    df_sampled = df_train.sample(
        n=train_size, weights=weights, replace=True, random_state=config.random_state
    ).reset_index(drop=True)
    logger.info(
        "Sampled %d training examples (density_col=%s, space=%s, k=%d)",
        train_size, density_col, space, k,
    )

    # ---- load model ----
    device_map = "auto"
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.unsloth_model,
        max_seq_length=config.max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=config.random_state,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # ---- format dataset ----
    conversations = _build_conversations(df_sampled)

    def _fmt(examples):
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for c in examples["conversations"]
        ]
        return {"text": texts}

    hf_dataset = HFDataset.from_dict({"conversations": conversations})
    hf_dataset = hf_dataset.map(_fmt, batched=True)

    # ---- output path ----
    model_slug = config.unsloth_model.replace("/", "_").replace(":", "_")
    density_tag = f"{'pca_' if space == 'pca' else ''}k{k}_ratio"
    out_dir = os.path.join(
        config.output_root, "llm_finetuned", model_slug,
        f"{dataset_tag}__{density_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model")

    # ---- train ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.llm_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=config.llm_max_steps,
            learning_rate=config.llm_learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=config.random_state,
            output_dir=os.path.join(out_dir, "checkpoints"),
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    logger.info("Starting LLM fine-tuning (max_steps=%d)...", config.llm_max_steps)
    trainer_stats = trainer.train()
    logger.info("Training complete. Loss=%.4f", trainer_stats.training_loss)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Model saved → %s", model_dir)

    # ---- evaluate on test set ----
    FastModel.for_inference(model)
    df_test = pd.read_csv(test_csv)
    test_texts  = df_test["text"].astype(str).tolist()
    test_labels = df_test["label"].astype(str).tolist()

    preds = []
    raw_responses = []

    logger.info("Running inference on %d test samples...", len(test_texts))
    for text_sample in test_texts:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"{_FT_CLASSIFY_INSTRUCTION}\n\nText: {text_sample}\n\nLabel:"}],
        }]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=16, use_cache=True, temperature=0.01,
            )
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        raw_responses.append(response)
        preds.append(_parse_ft_response(response))

    metrics = _compute_metrics(test_labels, preds)
    metrics["training_loss"] = float(trainer_stats.training_loss)
    metrics["density_col"]   = density_col
    metrics["k"]             = k
    metrics["space"]         = space

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = df_test[["text", "label"]].copy()
    pred_df["prediction"] = preds
    pred_df["raw_response"] = raw_responses
    pred_df["correct"] = pred_df["label"] == pred_df["prediction"]
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    logger.info(
        "LLM eval — F1=%.4f  Acc=%.4f  BalAcc=%.4f",
        metrics["f1"], metrics["accuracy"], metrics["balanced_accuracy"],
    )
    return metrics
