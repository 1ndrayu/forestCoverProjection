"""Utility script to engineer features from the base forest dataset.

Run this once locally (outside the Streamlit app):

    python data_processing.py

It will generate `processed_forest_data.csv`. Next, run
`python train_ann.py` to train the neural network and create
forecast files consumed by the Streamlit app.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
SOURCE_PATH = BASE_DIR / "annual-change-forest-area.csv"
PROCESSED_OUTPUT = BASE_DIR / "processed_forest_data.csv"

MIN_YEAR = 1990
MAX_YEAR = 2015
FORECAST_END_YEAR = 2030


def load_source() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Cannot find source CSV at {SOURCE_PATH}")

    df = pd.read_csv(SOURCE_PATH)
    df = df.rename(
        columns={
            "Entity": "entity",
            "Code": "code",
            "Year": "year",
            "Net forest conversion": "forest_change",
        }
    )

    df = df.dropna(subset=["forest_change"])
    df["forest_change"] = pd.to_numeric(df["forest_change"], errors="coerce")
    df = df.dropna(subset=["forest_change"])
    df["year"] = df["year"].astype(int)

    return df


def expand_and_engineer(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("year").set_index("year")

    full_years = pd.Index(range(MIN_YEAR, MAX_YEAR + 1), name="year")
    expanded = group.reindex(full_years)
    expanded["entity"] = group["entity"].iloc[0]
    expanded["code"] = group["code"].iloc[0]

    expanded["forest_change"] = expanded["forest_change"].interpolate(method="linear")
    expanded["forest_change"] = expanded["forest_change"].fillna(method="ffill").fillna(
        method="bfill"
    )

    expanded["forest_change"] = expanded["forest_change"].ffill().bfill()
    expanded["forest_change"] = expanded["forest_change"].astype(float)

    base_level = expanded["forest_change"].iloc[0]
    expanded["forest_cover_index"] = base_level + expanded["forest_change"].cumsum()

    expanded["cover_lag_1"] = expanded["forest_cover_index"].shift(1)
    expanded["cover_lag_2"] = expanded["forest_cover_index"].shift(2)
    expanded["cover_lag_3"] = expanded["forest_cover_index"].shift(3)

    expanded["change_lag_1"] = expanded["forest_change"].shift(1)
    expanded["change_lag_2"] = expanded["forest_change"].shift(2)

    expanded["year_norm"] = (expanded.index - MIN_YEAR) / (FORECAST_END_YEAR - MIN_YEAR)

    expanded["year"] = expanded.index
    return expanded.reset_index(drop=True)


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.groupby("entity", group_keys=False).apply(expand_and_engineer)
    processed = processed.sort_values(["entity", "year"]).reset_index(drop=True)
    return processed


def main() -> None:
    df = load_source()
    processed = build_processed_dataset(df)

    processed.to_csv(PROCESSED_OUTPUT, index=False)

    print(f"Saved processed data to {PROCESSED_OUTPUT}")
    print("Next, run `python train_ann.py` to train the ANN and generate forecasts.")


if __name__ == "__main__":
    main()
