"""Train an artificial neural network on the engineered forest dataset and
produce forecast CSVs consumed by the Streamlit app.

Usage (run after `python data_processing.py`):

    python train_ann.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
PROCESSED_PATH = BASE_DIR / "processed_forest_data.csv"
FORECAST_OUTPUT = BASE_DIR / "forest_forecast_annual.csv"
SUMMARY_OUTPUT = BASE_DIR / "forest_forecast_summary.csv"
METRICS_OUTPUT = BASE_DIR / "ann_training_metrics.json"

MIN_YEAR = 1990
MAX_YEAR = 2015
FORECAST_END_YEAR = 2030

FEATURE_COLUMNS: List[str] = [
    "forest_cover_index",
    "cover_lag_1",
    "cover_lag_2",
    "cover_lag_3",
    "change_lag_1",
    "change_lag_2",
    "forest_change",
    "year_norm",
]

COVER_FEATURES = [
    "forest_cover_index",
    "cover_lag_1",
    "cover_lag_2",
    "cover_lag_3",
]

CHANGE_FEATURES = ["change_lag_1", "change_lag_2", "forest_change"]


def load_processed_dataset() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
        )
    df = pd.read_csv(PROCESSED_PATH)
    return df


def _compute_entity_scales(group: pd.DataFrame) -> Tuple[float, float]:
    cover_scale = float(max(group["forest_cover_index"].abs().max(), 1.0))
    change_scale = float(max(group["forest_change"].abs().max(), 1.0))
    return cover_scale, change_scale


def linear_forecast_entity(
    years: np.ndarray, values: np.ndarray, future_years: np.ndarray
) -> np.ndarray:
    unique_years = np.unique(years)
    if len(unique_years) < 2:
        return np.repeat(values[-1], len(future_years))

    try:
        coeffs = np.polyfit(unique_years, values, deg=1)
        return np.polyval(coeffs, future_years)
    except np.linalg.LinAlgError:
        return np.repeat(values[-1], len(future_years))


def compute_blend_weight(group: pd.DataFrame) -> float:
    recent = group.sort_values("year").tail(5)
    diffs = recent["forest_cover_index"].diff().abs().dropna()
    volatility = float(diffs.mean()) if not diffs.empty else 0.0
    scale = float(max(recent["forest_cover_index"].abs().max(), 1.0))
    ratio = volatility / scale
    weight = 0.25 + ratio * 2.5
    return float(np.clip(weight, 0.3, 0.85))


def prepare_training_data(processed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    rows: List[np.ndarray] = []
    targets: List[float] = []
    entity_scales: Dict[str, Tuple[float, float]] = {}

    for entity, group in processed.groupby("entity", sort=False):
        group_sorted = group.sort_values("year").reset_index(drop=True)
        cover_scale, change_scale = _compute_entity_scales(group_sorted)
        entity_scales[entity] = (cover_scale, change_scale)

        for idx in range(len(group_sorted) - 1):
            current = group_sorted.iloc[idx]
            nxt = group_sorted.iloc[idx + 1]

            if current[FEATURE_COLUMNS].isna().any():
                continue
            if np.isnan(nxt["forest_cover_index"]):
                continue

            feature_series = current[FEATURE_COLUMNS].fillna(0.0).copy()
            feature_series[COVER_FEATURES] = feature_series[COVER_FEATURES] / cover_scale
            feature_series[CHANGE_FEATURES] = feature_series[CHANGE_FEATURES] / change_scale

            rows.append(feature_series.to_numpy(dtype=float))
            targets.append(float(nxt["forest_cover_index"] / cover_scale))

    if not rows:
        raise RuntimeError("Insufficient data to assemble training samples. Check the processed CSV.")

    X = np.vstack(rows)
    y = np.asarray(targets, dtype=float)
    return X, y, entity_scales


def train_model(X: np.ndarray, y: np.ndarray):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)

    scaler_X = StandardScaler()
    scaler_X.fit(X_train)

    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        random_state=42,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    model.fit(X_train_scaled, y_train_scaled)

    train_pred = scaler_y.inverse_transform(
        model.predict(X_train_scaled).reshape(-1, 1)
    ).ravel()
    val_pred = scaler_y.inverse_transform(
        model.predict(X_val_scaled).reshape(-1, 1)
    ).ravel()

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "val_r2": float(r2_score(y_val, val_pred)),
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
    }

    return model, scaler_X, scaler_y, metrics


def _prepare_next_row(current: pd.Series, predicted_cover: float) -> pd.Series:
    next_year = int(current["year"]) + 1
    next_row = current.copy()
    next_row["year"] = next_year
    next_row["forest_cover_index"] = predicted_cover
    next_row["forest_change"] = predicted_cover - current["forest_cover_index"]
    next_row["cover_lag_1"] = current["forest_cover_index"]
    next_row["cover_lag_2"] = current["cover_lag_1"]
    next_row["cover_lag_3"] = current["cover_lag_2"]
    next_row["change_lag_1"] = current["forest_change"]
    next_row["change_lag_2"] = current["change_lag_1"]
    next_row["year_norm"] = (next_year - MIN_YEAR) / (FORECAST_END_YEAR - MIN_YEAR)
    return next_row


def _scale_features(row: pd.Series, cover_scale: float, change_scale: float) -> np.ndarray:
    feature_series = row[FEATURE_COLUMNS].fillna(0.0).copy()
    feature_series[COVER_FEATURES] = feature_series[COVER_FEATURES] / cover_scale
    feature_series[CHANGE_FEATURES] = feature_series[CHANGE_FEATURES] / change_scale
    return feature_series.to_numpy(dtype=float)


def generate_ann_forecasts(
    processed: pd.DataFrame,
    model: MLPRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    entity_scales: Dict[str, Tuple[float, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    forecast_records: List[dict] = []
    summary_records: List[dict] = []
    entity_weights: Dict[str, float] = {}

    for entity, group in processed.groupby("entity", sort=False):
        group_sorted = group.sort_values("year").reset_index(drop=True)
        group_sorted[FEATURE_COLUMNS] = group_sorted[FEATURE_COLUMNS].fillna(0.0)
        code = group_sorted["code"].iloc[0]
        working = group_sorted.copy()

        cover_scale, change_scale = entity_scales.get(entity, _compute_entity_scales(group_sorted))
        final_value = float(working.iloc[-1]["forest_cover_index"])
        historical_years = group_sorted["year"].to_numpy(dtype=float)
        historical_values = group_sorted["forest_cover_index"].to_numpy(dtype=float)
        future_years = np.arange(MAX_YEAR + 1, FORECAST_END_YEAR + 1, dtype=int)
        linear_preds = linear_forecast_entity(historical_years, historical_values, future_years)
        blend_weight = compute_blend_weight(group_sorted)
        entity_weights[entity] = blend_weight

        for step, year in enumerate(future_years):
            current_row = working.iloc[-1]
            scaled_features = _scale_features(current_row, cover_scale, change_scale)
            pred_scaled = model.predict(scaler_X.transform([scaled_features]))[0]
            predicted_cover_norm = scaler_y.inverse_transform(
                np.array(pred_scaled).reshape(1, -1)
            )[0][0]
            ann_cover = float(predicted_cover_norm * cover_scale)
            linear_cover = float(linear_preds[step])
            blended_cover = blend_weight * ann_cover + (1 - blend_weight) * linear_cover

            next_year = int(year)
            forecast_records.append(
                {
                    "entity": entity,
                    "code": code,
                    "year": next_year,
                    "predicted_forest_cover": blended_cover,
                }
            )

            final_value = blended_cover
            next_row = _prepare_next_row(current_row, blended_cover)
            working = pd.concat([working, next_row.to_frame().T], ignore_index=True)

            cover_scale = float(max(cover_scale, abs(blended_cover), 1.0))
            change_scale = float(
                max(change_scale, abs(next_row["forest_change"]) if "forest_change" in next_row else 0.0, 1.0)
            )

        summary_records.append(
            {
                "entity": entity,
                "code": code,
                "year": FORECAST_END_YEAR,
                "predicted_forest_cover": final_value,
            }
        )

    forecast_df = pd.DataFrame(forecast_records)
    summary_df = pd.DataFrame(summary_records)
    return forecast_df, summary_df, entity_weights


def main() -> None:
    processed = load_processed_dataset()

    X, y, entity_scales = prepare_training_data(processed)
    model, scaler_X, scaler_y, metrics = train_model(X, y)
    forecast_df, summary_df, entity_weights = generate_ann_forecasts(
        processed, model, scaler_X, scaler_y, entity_scales
    )

    forecast_df.to_csv(FORECAST_OUTPUT, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    try:
        import json

        metrics["blend_weight_mean"] = float(np.mean(list(entity_weights.values()))) if entity_weights else None
        metrics["blend_weights"] = {k: float(v) for k, v in entity_weights.items()}

        with METRICS_OUTPUT.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
    except Exception:
        pass

    print("ANN training complete.")
    print(f"Train MAE: {metrics['train_mae']:.3f} | Val MAE: {metrics['val_mae']:.3f}")
    print(f"Train R^2: {metrics['train_r2']:.3f} | Val R^2: {metrics['val_r2']:.3f}")
    print(f"Saved annual forecasts to {FORECAST_OUTPUT}")
    print(f"Saved summary forecasts to {SUMMARY_OUTPUT}")
    if METRICS_OUTPUT.exists():
        print(f"Training metrics written to {METRICS_OUTPUT}")

if __name__ == "__main__":
    main()