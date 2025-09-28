"""Backend helpers for loading processed forest datasets and building views."""

from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).parent
PROCESSED_PATH = BASE_DIR / "processed_forest_data.csv"
FORECAST_PATH = BASE_DIR / "forest_forecast_annual.csv"
SUMMARY_PATH = BASE_DIR / "forest_forecast_summary.csv"


def _ensure_data_files() -> None:
    missing = [path.name for path in (PROCESSED_PATH, FORECAST_PATH, SUMMARY_PATH) if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Missing processed dataset(s): "
            f"{joined}. Run `python data_processing.py` to generate required files."
        )


def load_processed_data() -> pd.DataFrame:
    _ensure_data_files()
    df = pd.read_csv(PROCESSED_PATH)
    df["year"] = df["year"].astype(int)
    return df


def load_forecast_data() -> pd.DataFrame:
    _ensure_data_files()
    df = pd.read_csv(FORECAST_PATH)
    df["year"] = df["year"].astype(int)
    return df


def load_summary_data() -> pd.DataFrame:
    _ensure_data_files()
    df = pd.read_csv(SUMMARY_PATH)
    df["year"] = df["year"].astype(int)
    return df


def list_entities(processed: pd.DataFrame) -> List[str]:
    entities = processed["entity"].dropna().unique().tolist()
    entities.sort()
    return entities


def select_default_entity(entities: List[str]) -> str:
    if not entities:
        raise ValueError("No entities available in processed dataset")
    return "World" if "World" in entities else entities[0]


def build_timeseries(entity: str, processed: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    historical = (
        processed[processed["entity"] == entity][["year", "forest_cover_index"]]
        .rename(columns={"forest_cover_index": "forest_cover"})
        .assign(phase="Historical")
    )

    future = (
        forecast[forecast["entity"] == entity][["year", "predicted_forest_cover"]]
        .rename(columns={"predicted_forest_cover": "forest_cover"})
        .assign(phase="Forecast")
    )

    combined = pd.concat([historical, future], ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)

    if combined.empty:
        raise ValueError(f"No data available for entity '{entity}'.")

    return combined
