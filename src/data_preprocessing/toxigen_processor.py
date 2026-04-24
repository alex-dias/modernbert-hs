"""
ToxiGen dataset processor.

ToxiGen raw files are expected at configs/datasets.yaml `toxigen.data_dir`.
One CSV per target group, named {group}.csv, with columns [text, label]
where label is "hate" or "no hate" (space — normalized to underscore here).

Supported groups (from original dataHandler.py):
    asian, black, chinese, jewish, latino, lgbtq, mental_dis, mexican,
    middle_east, muslim, native_american, physical_dis, women
"""

import os
import pandas as pd
from .base_processor import BaseProcessor

TOXIGEN_GROUPS = [
    "asian", "black", "chinese", "jewish", "latino", "lgbtq",
    "mental_dis", "mexican", "middle_east", "muslim",
    "native_american", "physical_dis", "women",
]

LABEL_MAP = {
    "hate": "hate",
    "no hate": "no_hate",
    "no_hate": "no_hate",
    "0": "no_hate",
    "1": "hate",
}


class ToxigenProcessor(BaseProcessor):
    """
    Processes the full ToxiGen dataset (all groups combined) or a single group.

    Args:
        data_dir:  path to folder containing {group}.csv files
        groups:    list of groups to include; defaults to all TOXIGEN_GROUPS
        output_root / test_size / random_state: passed to BaseProcessor
    """

    def __init__(
        self,
        data_dir: str,
        groups: list[str] | None = None,
        output_root: str = "outputs/1_preprocessed",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__("toxigen", output_root, test_size, random_state)
        self.data_dir = data_dir
        self.groups = groups or TOXIGEN_GROUPS

    def load_raw(self) -> pd.DataFrame:
        dfs = []
        for group in self.groups:
            path = os.path.join(self.data_dir, f"{group}.csv")
            if not os.path.exists(path):
                print(f"  [ToxiGen] Warning: {path} not found — skipping {group}")
                continue
            df = pd.read_csv(path)
            df = self._normalize_columns(df, group)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No ToxiGen CSVs found in {self.data_dir}")

        return pd.concat(dfs, ignore_index=True)

    def _normalize_columns(self, df: pd.DataFrame, group: str) -> pd.DataFrame:
        df = df.copy()

        # Flexible column name matching
        col_map = {c.lower().strip(): c for c in df.columns}

        text_col = col_map.get("text") or col_map.get("tweet") or col_map.get("tweet treated")
        label_col = col_map.get("label") or col_map.get("class")

        if text_col is None or label_col is None:
            raise ValueError(f"[ToxiGen/{group}] Cannot find text/label columns. Found: {list(df.columns)}")

        out = pd.DataFrame({
            "text": df[text_col],
            "group": group,
            "label": df[label_col].astype(str).str.strip().str.lower().map(LABEL_MAP),
        })

        n_unmapped = out["label"].isna().sum()
        if n_unmapped > 0:
            print(f"  [ToxiGen/{group}] Warning: {n_unmapped} rows had unrecognized labels — dropping")
            out = out.dropna(subset=["label"])

        return out
