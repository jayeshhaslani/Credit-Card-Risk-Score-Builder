from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np

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


def _apply_woebin_ply(
    applicant_df: pd.DataFrame,
    bins: dict[str, pd.DataFrame],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Lightweight replacement for scorecardpy.woebin_ply that applies saved WOE bins.

    Produces columns named `<feature>_woe` for each feature present in `bins` and
    returns them in the same order as the trained model expects.
    """
    result = pd.DataFrame(index=applicant_df.index)

    for feature, bin_df in bins.items():
        if feature not in applicant_df.columns:
            continue

        series = applicant_df[feature]

        # Attempt to interpret breaks as numeric cut points
        breaks = list(bin_df["breaks"].tolist())
        numeric_breaks = []
        is_numeric = True
        for b in breaks:
            try:
                if str(b) == "inf":
                    numeric_breaks.append(np.inf)
                else:
                    numeric_breaks.append(float(b))
            except Exception:
                is_numeric = False
                break

        woe_values = []
        if is_numeric:
            # edges: [-inf, b1, b2, ..., inf]
            edges = np.array([-np.inf] + numeric_breaks, dtype=float)
            uppers = edges[1:]

            for v in series:
                if pd.isna(v):
                    woe_values.append(0.0)
                    continue
                try:
                    fv = float(v)
                except Exception:
                    woe_values.append(0.0)
                    continue
                # searchsorted with side='right' matches binning semantics like [a,b)
                idx = np.searchsorted(uppers, fv, side="right")
                # guard index range
                if idx < 0:
                    idx = 0
                if idx >= len(bin_df):
                    idx = len(bin_df) - 1
                woe_values.append(float(bin_df.iloc[int(idx)]["woe"]))
        else:
            # Categorical / string bins. `breaks` may contain comma-separated tokens
            patterns = []
            for _, row in bin_df.iterrows():
                token = str(row["breaks"])
                parts = [p for p in token.split(",") if p != ""]
                patterns.append((parts, float(row["woe"])))

            def match_cat(val):
                if pd.isna(val):
                    return 0.0
                s = str(val)
                for parts, w in patterns:
                    for p in parts:
                        if "%" in p:
                            # treat % as wildcard (substring match)
                            sub = p.replace("%", "")
                            if sub == "":
                                return w
                            if sub in s:
                                return w
                        else:
                            if s == p:
                                return w
                return 0.0

            woe_values = [match_cat(v) for v in series]

        result[f"{feature}_woe"] = pd.Series(woe_values, index=applicant_df.index, dtype=float)

    if expected_columns:
        return result.reindex(columns=expected_columns, fill_value=0.0)

    return result


def predict_applicant(raw_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the applicant frame, transform it with WOE bins, and return a prediction payload."""

    model, bins, feature_names = _load_artifacts()
    applicant_df = build_applicant_dataframe(raw_inputs)

    missing_columns = [feature for feature in feature_names if feature not in applicant_df.columns]
    if missing_columns:
        raise ValueError(f"Applicant dataframe is missing required feature columns: {missing_columns}")

    applicant_df = applicant_df.loc[:, feature_names]

    expected_model_columns = list(getattr(model, "feature_names_in_", []))
    if not expected_model_columns:
        expected_model_columns = [f"{feature}_woe" for feature in feature_names]

    try:
        woe_df = _apply_woebin_ply(applicant_df, bins, expected_columns=expected_model_columns)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise RuntimeError(f"WOE transformation failed for applicant input: {exc}") from exc

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
