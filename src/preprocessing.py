from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


DEFAULT_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "revol_bal",
    "revol_util",
    "delinq_2yrs",
    "pub_rec",
    "inq_last_6mths",
    "unrate",
    "fedfunds",
    "cpi",
]

DEFAULTS: dict[str, Any] = {
    "loan_amnt": 10000.0,
    "term": "36 months",
    "int_rate": 10.0,
    "grade": "B",
    "sub_grade": "B1",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 60000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 15.0,
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "open_acc": 10,
    "revol_bal": 10000.0,
    "revol_util": 45.0,
    "delinq_2yrs": 0,
    "pub_rec": 0,
    "inq_last_6mths": 1,
    "unrate": 4.2,
    "fedfunds": 4.5,
    "cpi": 320.0,
}

ALLOWED_VALUES: dict[str, list[str]] = {
    "term": ["36 months", "60 months"],
    "home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"],
    "verification_status": ["Verified", "Source Verified", "Not Verified"],
    "grade": ["A", "B", "C", "D", "E", "F", "G"],
    "purpose": [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "small_business",
        "car",
        "medical",
        "moving",
        "vacation",
        "wedding",
        "house",
        "educational",
        "renewable_energy",
        "other",
    ],
}


def _feature_order() -> list[str]:
    model_features_path = Path(__file__).resolve().parents[1] / "models" / "model_features.pkl"
    if model_features_path.exists():
        try:
            features = joblib.load(model_features_path)
            if isinstance(features, list):
                return [str(feature) for feature in features]
        except Exception:
            pass
    return DEFAULT_FEATURES


def _coerce_value(column: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return DEFAULTS[column]

    if column in {"loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high", "revol_bal", "revol_util", "int_rate", "open_acc", "delinq_2yrs", "pub_rec", "inq_last_6mths", "unrate", "fedfunds", "cpi"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(DEFAULTS[column])

    if column == "term":
        return f" {str(value).strip()}"

    if column in {"grade", "sub_grade", "home_ownership", "verification_status", "purpose", "emp_length"}:
        cleaned = str(value).strip()
        if not cleaned:
            cleaned = DEFAULTS[column]
        if column == "grade":
            cleaned = cleaned.upper()
        elif column == "verification_status":
            cleaned = cleaned.title()
        elif column == "purpose":
            cleaned = cleaned.lower()
        return cleaned

    return value


def _validate_value(column: str, value: Any) -> None:
    allowed = ALLOWED_VALUES.get(column)
    if allowed is None:
        return
    if str(value).strip() not in allowed:
        raise ValueError(f"Unsupported value for {column}: {value}. Expected one of {allowed}.")


def build_applicant_dataframe(raw_inputs: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Create a single-row applicant DataFrame with the same schema expected by the trained model."""

    feature_names = _feature_order()
    normalized_inputs = dict(raw_inputs or {})

    row: dict[str, Any] = {}
    for column in feature_names:
        raw_value = normalized_inputs.get(column)
        value = _coerce_value(column, raw_value)
        if value is None:
            value = DEFAULTS[column]
        if column in {"grade", "home_ownership", "verification_status", "purpose", "emp_length", "sub_grade"}:
            value = str(value).strip()
            if column == "grade":
                value = value.upper()
            if column == "verification_status":
                value = value.title()
            if column == "purpose":
                value = value.lower()
        if column == "term":
            value = f" {str(value).strip()}"
        _validate_value(column, value)
        row[column] = value

    dataframe = pd.DataFrame([row], columns=feature_names)
    return dataframe
