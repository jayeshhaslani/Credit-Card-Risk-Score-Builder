from __future__ import annotations

import joblib
import streamlit as st

from apps.utils.ui_helpers import render_sidebar
from src.predictor import artifact_path, predict_applicant
from src.scoring import score_probability


def render() -> None:
    st.title("Risk Simulator")
    st.caption("Enter applicant and macroeconomic inputs to generate a real-time credit decision.")
    st.info("This workflow uses the existing production model artifacts and WOE transformation pipeline.")

    inputs = render_sidebar()

    if st.button("Run Prediction", type="primary"):
        try:
            prediction = predict_applicant(inputs)
            scoring = score_probability(prediction["probability_of_default"])
        except FileNotFoundError as exc:
            st.error(f"Model artifacts are missing: {exc}")
            st.stop()
        except ValueError as exc:
            st.error(f"Input validation failed: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"Prediction could not be completed: {exc}")
            st.stop()
        except Exception:
            st.error("Prediction service is temporarily unavailable. Please try again later.")
            st.stop()

        st.subheader("Prediction Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Credit Score", scoring["credit_score"])
        col2.metric("Probability of Default", f"{scoring['probability_of_default']:.2%}")
        col3.metric("Risk Band", scoring["risk_band"])
        col4.metric("Decision", scoring["decision"])

        st.subheader("Model Drivers")
        st.caption("The most influential features are shown using the logistic regression coefficients.")
        try:
            model = joblib.load(artifact_path("macro_logistic_model.pkl"))
        except Exception:
            model = None

        if model is not None and hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            feature_names = list(getattr(model, "feature_names_in_", []))
            driver_frame = {"feature": feature_names, "coefficient": coefficients}
            st.dataframe(driver_frame, use_container_width=True)
        else:
            st.info("Model coefficients were not available, so the driver explanation is unavailable.")
