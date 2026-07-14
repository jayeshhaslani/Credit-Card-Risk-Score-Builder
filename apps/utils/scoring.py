import os

# Defer heavy imports and artifact loading until scoring is requested
_MODEL = None
_BINS = None


def _load_artifacts():
    global _MODEL, _BINS
    if _MODEL is None or _BINS is None:
        import joblib
        # scorecardpy is not required at runtime for inference; use saved bins directly

        # resolve models directory relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(project_root, "..", "models")
        models_dir = os.path.normpath(models_dir)

        _MODEL = joblib.load(os.path.join(models_dir, "macro_logistic_model.pkl"))
        _BINS = joblib.load(os.path.join(models_dir, "macro_woe_bins.pkl"))


def score_applicant(input_df):
    """Score a single applicant dataframe (expects 1-row DataFrame).

    Returns dict: {"pd": float, "score": int, "decision": str}
    """
    _load_artifacts()

    from src.predictor import _apply_woebin_ply

    # Remove target/label columns if present (model was trained without them)
    df_features = input_df.drop(columns=['target', 'loan_status'], errors='ignore')

    input_woe = _apply_woebin_ply(df_features, _BINS)

    expected_model_columns = list(getattr(_MODEL, "feature_names_in_", []))
    if not expected_model_columns:
        # derive from bins ordering
        expected_model_columns = [f"{feature}_woe" for feature in (_BINS.keys())]

    input_woe = input_woe.reindex(columns=expected_model_columns, fill_value=0.0)

    pd_value = _MODEL.predict_proba(input_woe)[:, 1][0]

    credit_score = int(round(850 - (pd_value * 550)))

    if pd_value < 0.10:
        decision = "Approve"
    elif pd_value < 0.25:
        decision = "Manual Review"
    else:
        decision = "Reject"

    return {"pd": pd_value, "score": credit_score, "decision": decision}