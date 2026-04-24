"""
Base dataset processor.

Defines the standard schema and shared preprocessing logic.
All dataset-specific processors should subclass BaseProcessor,
implement `load_raw()`, and call `process()`.

Output schema per sample:
    text  (str)   — raw text
    group (str)   — target group identifier (e.g. "russian", "jewish")
    label (str)   — "hate" or "no_hate"
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"text", "group", "label"}
VALID_LABELS = {"hate", "no_hate"}


class BaseProcessor:
    """
    Shared preprocessing logic for all datasets.

    Subclasses must implement:
        load_raw() -> pd.DataFrame  with at minimum columns [text, group, label]
    """

    def __init__(self, dataset_name: str, output_root: str = "outputs/1_preprocessed", test_size: float = 0.2, random_state: int = 42):
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_root, dataset_name)
        self.test_size = test_size
        self.random_state = random_state

    def load_raw(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement load_raw()")

    def _validate(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"[{self.dataset_name}] Missing columns after load_raw(): {missing}")
        invalid = set(df["label"].unique()) - VALID_LABELS
        if invalid:
            raise ValueError(f"[{self.dataset_name}] Invalid label values: {invalid}. Expected: {VALID_LABELS}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[list(REQUIRED_COLUMNS)].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["group"] = df["group"].astype(str).str.strip().str.lower()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"] != ""]
        return df.reset_index(drop=True)

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Stratify by label to preserve class balance
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["label"],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def process(self) -> dict[str, pd.DataFrame]:
        """
        Full pipeline: load → validate → clean → split → save.
        Returns {"full": df, "train": df_train, "test": df_test}.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        df = self.load_raw()
        self._validate(df)
        df = self._clean(df)

        train, test = self._split(df)

        df.to_csv(os.path.join(self.output_dir, "full.csv"), index=False)
        train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        print(f"[{self.dataset_name}] Saved {len(df)} samples → {self.output_dir}")
        print(f"  Train: {len(train)} | Test: {len(test)}")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
        print(f"  Groups: {sorted(df['group'].unique().tolist())}")

        return {"full": df, "train": train, "test": test}
