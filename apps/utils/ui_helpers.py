from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.fred_client import get_macro_indicators


PREDICTION_FIELDS = [
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
]


def render_sidebar() -> dict[str, Any]:
    """Render the shared prediction input sidebar with sensible defaults."""

    macro_defaults = get_macro_indicators()

    st.sidebar.header("Applicant Information")
    st.sidebar.caption("Adjust the applicant and macro inputs before scoring.")
    annual_inc = st.sidebar.number_input("Annual Income", min_value=0.0, value=60000.0, step=1000.0)
    emp_length = st.sidebar.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verification_status = st.sidebar.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

    st.sidebar.header("Loan Information")
    loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0.0, value=10000.0, step=1000.0)
    term = st.sidebar.selectbox("Loan Term", ["36 months", "60 months"])
    purpose = st.sidebar.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "medical", "moving", "vacation", "wedding", "house", "educational", "renewable_energy"])
    interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    st.sidebar.header("Credit Profile")
    fico_score = st.sidebar.number_input("FICO Score", min_value=300, max_value=850, value=700)
    revolving_balance = st.sidebar.number_input("Revolving Balance", min_value=0.0, value=10000.0, step=1000.0)
    revolving_utilization = st.sidebar.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
    open_accounts = st.sidebar.number_input("Open Accounts", min_value=0, value=10, step=1)
    public_records = st.sidebar.number_input("Public Records", min_value=0, value=0, step=1)
    delinquencies = st.sidebar.number_input("Delinquencies", min_value=0, value=0, step=1)
    recent_inquiries = st.sidebar.number_input("Recent Inquiries", min_value=0, value=1, step=1)
    dti = st.sidebar.number_input("Debt To Income Ratio", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

    st.sidebar.header("Macro Conditions")
    use_manual_macro = st.sidebar.checkbox("Manually override FRED values", value=False)
    unrate = st.sidebar.number_input("Unemployment Rate", min_value=0.0, max_value=100.0, value=float(macro_defaults.get("unrate", 4.2)), step=0.1, disabled=not use_manual_macro)
    fedfunds = st.sidebar.number_input("Federal Funds Rate", min_value=-10.0, max_value=100.0, value=float(macro_defaults.get("fedfunds", 4.5)), step=0.1, disabled=not use_manual_macro)
    cpi = st.sidebar.number_input("CPI", min_value=0.0, value=float(macro_defaults.get("cpi", 320.0)), step=1.0, disabled=not use_manual_macro)

    return {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": interest_rate,
        "grade": "B",
        "sub_grade": "B1",
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "fico_range_low": float(fico_score),
        "fico_range_high": float(fico_score),
        "open_acc": open_accounts,
        "revol_bal": revolving_balance,
        "revol_util": revolving_utilization,
        "delinq_2yrs": delinquencies,
        "pub_rec": public_records,
        "inq_last_6mths": recent_inquiries,
        "unrate": unrate,
        "fedfunds": fedfunds,
        "cpi": cpi,
    }


def explain_top_drivers(model_coefficients: pd.Series | None = None) -> list[tuple[str, float]]:
    if model_coefficients is None:
        return []
    ranked = sorted(model_coefficients.items(), key=lambda item: abs(float(item[1])), reverse=True)
    return [(name, float(value)) for name, value in ranked[:5]]
