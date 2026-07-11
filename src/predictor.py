from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import scorecardpy as sc

from src.preprocessing import build_applicant_dataframe


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_path(name: str) -> Path:
    return _project_root() / "models" / name


def _artifact_path(name: str) -> Path:
    return artifact_path(name)


def _load_artifacts() -> tuple[Any, Any, list[str]]:
    model_path = _artifact_path("macro_logistic_model.pkl")
    bins_path = _artifact_path("macro_woe_bins.pkl")
    features_path = _artifact_path("model_features.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not bins_path.exists():
        raise FileNotFoundError(f"Missing WOE bins artifact: {bins_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature-order artifact: {features_path}")

    model = joblib.load(model_path)
    bins = joblib.load(bins_path)
    feature_names = joblib.load(features_path)
    return model, bins, list(feature_names)


def predict_applicant(raw_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the applicant frame, transform it with WOE bins, and return a prediction payload."""

    model, bins, feature_names = _load_artifacts()
    applicant_df = build_applicant_dataframe(raw_inputs)

    missing_columns = [feature for feature in feature_names if feature not in applicant_df.columns]
    if missing_columns:
        raise ValueError(f"Applicant dataframe is missing required feature columns: {missing_columns}")

    applicant_df = applicant_df.loc[:, feature_names]

    try:
        woe_df = sc.woebin_ply(applicant_df, bins)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise RuntimeError(f"WOE transformation failed for applicant input: {exc}") from exc

    expected_model_columns = list(getattr(model, "feature_names_in_", []))
    if not expected_model_columns:
        expected_model_columns = [f"{feature}_woe" for feature in feature_names]

    if not all(column in woe_df.columns for column in expected_model_columns):
        missing_woe_columns = [column for column in expected_model_columns if column not in woe_df.columns]
        raise ValueError(f"WOE transformation did not produce the expected feature set. Missing: {missing_woe_columns}")

    woe_df = woe_df.reindex(columns=expected_model_columns, fill_value=0.0)

    probability = float(model.predict_proba(woe_df)[0, 1])

    return {
        "probability_of_default": probability,
        "raw_dataframe": applicant_df,
        "woe_dataframe": woe_df,
    }
