"""
Russian hate speech dataset processor.

The Russian dataset has two source files:
    annotated_path  — russ_annot_masked.csv: labeled samples (text, label)
                      Used for supervised training and evaluation.
    full_path       — completeDataset_inference.csv: full unlabeled corpus
                      Column "Tweet Treated" contains the raw text.
                      Used for embedding generation only (no label → not included
                      in the standard preprocessed output).

The processor outputs only the annotated split under outputs/1_preprocessed/russian/.
The full corpus path is stored as an attribute for downstream embedding steps.
"""

import os
import pandas as pd
from .base_processor import BaseProcessor

LABEL_MAP = {
    "hate": "hate",
    "no hate": "no_hate",
    "no_hate": "no_hate",
    "0": "no_hate",
    "1": "hate",
}


class RussianProcessor(BaseProcessor):
    """
    Processes the Russian hate speech dataset.

    Args:
        annotated_path: path to the annotated CSV (labeled samples)
        full_corpus_path: optional path to the full unlabeled corpus CSV
        output_root / test_size / random_state: passed to BaseProcessor
    """

    def __init__(
        self,
        annotated_path: str,
        full_corpus_path: str | None = None,
        output_root: str = "outputs/1_preprocessed",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__("russian", output_root, test_size, random_state)
        self.annotated_path = annotated_path
        self.full_corpus_path = full_corpus_path

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.annotated_path):
            raise FileNotFoundError(f"[Russian] Annotated file not found: {self.annotated_path}")

        df = pd.read_csv(self.annotated_path)
        col_map = {c.lower().strip(): c for c in df.columns}

        text_col = col_map.get("text") or col_map.get("tweet treated") or col_map.get("tweet")
        label_col = col_map.get("label") or col_map.get("class")

        if text_col is None or label_col is None:
            raise ValueError(f"[Russian] Cannot find text/label columns. Found: {list(df.columns)}")

        out = pd.DataFrame({
            "text": df[text_col],
            "group": "russian",
            "label": df[label_col].astype(str).str.strip().str.lower().map(LABEL_MAP),
        })

        n_unmapped = out["label"].isna().sum()
        if n_unmapped > 0:
            print(f"  [Russian] Warning: {n_unmapped} rows had unrecognized labels — dropping")
            out = out.dropna(subset=["label"])

        return out

    def load_full_corpus_df(self) -> pd.DataFrame:
        """
        Load the full unlabeled Russian corpus as a DataFrame, preserving all
        original columns (Date, Tweet Treated, Tweet Raw, Url, Id) and adding
        a normalised 'text' column for inference.
        """
        if self.full_corpus_path is None:
            raise ValueError("[Russian] full_corpus_path was not provided.")
        if not os.path.exists(self.full_corpus_path):
            raise FileNotFoundError(f"[Russian] Full corpus not found: {self.full_corpus_path}")

        df = pd.read_csv(self.full_corpus_path)
        col_map = {c.lower().strip(): c for c in df.columns}
        text_col = col_map.get("tweet treated") or col_map.get("text") or col_map.get("tweet")

        if text_col is None:
            raise ValueError(f"[Russian] Cannot find text column in full corpus. Found: {list(df.columns)}")

        df["text"] = df[text_col].astype(str).str.strip()
        print(f"  [Russian] Loaded {len(df)} rows from full corpus")
        return df

    def load_full_corpus(self) -> list[str]:
        """
        Load the full unlabeled Russian corpus for embedding generation.
        Returns a list of raw text strings.
        """
        if self.full_corpus_path is None:
            raise ValueError("[Russian] full_corpus_path was not provided.")
        if not os.path.exists(self.full_corpus_path):
            raise FileNotFoundError(f"[Russian] Full corpus not found: {self.full_corpus_path}")

        df = pd.read_csv(self.full_corpus_path)
        col_map = {c.lower().strip(): c for c in df.columns}
        text_col = col_map.get("tweet treated") or col_map.get("text") or col_map.get("tweet")

        if text_col is None:
            raise ValueError(f"[Russian] Cannot find text column in full corpus. Found: {list(df.columns)}")

        texts = df[text_col].astype(str).str.strip().dropna().tolist()
        print(f"  [Russian] Loaded {len(texts)} texts from full corpus")
        return texts
