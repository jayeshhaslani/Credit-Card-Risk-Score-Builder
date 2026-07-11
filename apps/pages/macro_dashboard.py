from __future__ import annotations

import streamlit as st

from src.fred_client import get_macro_history, get_macro_indicators


def render() -> None:
    st.title("Macroeconomic Dashboard")
    st.caption("Monitor the latest macroeconomic conditions that influence the scorecard.")

    indicators = get_macro_indicators()
    if indicators.get("_source") != "fred":
        st.warning("FRED data could not be fetched; using fallback macro values for the dashboard.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unemployment Rate", f"{indicators['unrate']:.2f}%")
    col2.metric("Federal Funds Rate", f"{indicators['fedfunds']:.2f}%")
    col3.metric("CPI", f"{indicators['cpi']:.2f}")

    history = get_macro_history(periods=24)
    if not history.empty:
        st.subheader("Recent Trend")
        st.line_chart(history[["unrate", "fedfunds", "cpi"]])
    else:
        st.info("Macro history is currently unavailable; default values are being shown.")
