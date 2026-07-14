import joblib
import pandas as pd
import scorecardpy as sc

from src.predictor import _apply_woebin_ply, _artifact_path, predict_applicant
from src.preprocessing import build_applicant_dataframe


def _sample_inputs() -> dict[str, object]:
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 10,
        "grade": "B",
        "sub_grade": "B1",
        "emp_length": "10+ years",
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 15,
        "fico_range_low": 700,
        "fico_range_high": 704,
        "open_acc": 10,
        "revol_bal": 10000,
        "revol_util": 45,
        "delinq_2yrs": 0,
        "pub_rec": 0,
        "inq_last_6mths": 1,
        "unrate": 4.2,
        "fedfunds": 4.5,
        "cpi": 320,
    }


def test_apply_woebin_ply_respects_model_feature_order_and_dtype() -> None:
    model = joblib.load(_artifact_path("macro_logistic_model.pkl"))
    bins = joblib.load(_artifact_path("macro_woe_bins.pkl"))
    feature_names = joblib.load(_artifact_path("model_features.pkl"))

    applicant_df = build_applicant_dataframe(_sample_inputs()).loc[:, feature_names]
    expected_columns = list(getattr(model, "feature_names_in_", []))

    woe_df = _apply_woebin_ply(applicant_df, bins, expected_columns=expected_columns)

    assert list(woe_df.columns) == expected_columns
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in woe_df.dtypes)
    assert not any(pd.api.types.is_object_dtype(dtype) for dtype in woe_df.dtypes)
    assert not any(pd.api.types.is_categorical_dtype(dtype) for dtype in woe_df.dtypes)


def test_predict_applicant_matches_scorecardpy_woe_output() -> None:
    bins = joblib.load(_artifact_path("macro_woe_bins.pkl"))
    model = joblib.load(_artifact_path("macro_logistic_model.pkl"))
    feature_names = joblib.load(_artifact_path("model_features.pkl"))
    expected_columns = list(getattr(model, "feature_names_in_", []))

    applicant_df = build_applicant_dataframe(_sample_inputs()).loc[:, feature_names]
    custom_woe = _apply_woebin_ply(applicant_df, bins, expected_columns=expected_columns)
    scorecard_woe = sc.woebin_ply(applicant_df, bins).reindex(columns=expected_columns, fill_value=0.0)

    pd.testing.assert_frame_equal(
        custom_woe,
        scorecard_woe,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
        check_dtype=False,
    )


def test_predict_applicant_returns_probability() -> None:
    prediction = predict_applicant(_sample_inputs())

    assert prediction["probability_of_default"] > 0.0
    assert prediction["probability_of_default"] < 1.0
    assert pd.notna(prediction["probability_of_default"])
