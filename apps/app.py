from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.pages.macro_dashboard import render as render_macro_dashboard
from apps.pages.risk_simulator import render as render_risk_simulator
from apps.pages.regulatory_assistant import render as render_regulatory_assistant


def render() -> None:
    st.set_page_config(
        page_title="Credit Risk Intelligence Platform",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    st.sidebar.caption("Switch between the simulator, macro dashboard, and policy assistant.")
    selected_page = st.sidebar.radio(
        "Choose a page",
        ["Risk Simulator", "Macro Dashboard", "Regulatory Assistant"],
    )

    st.title("Credit Risk Intelligence Platform")
    st.markdown("A production-ready credit risk workspace for scoring, monitoring, and compliance workflows.")

    if selected_page == "Risk Simulator":
        render_risk_simulator()
    elif selected_page == "Macro Dashboard":
        render_macro_dashboard()
    else:
        render_regulatory_assistant()


def main() -> None:
    render()


if __name__ == "__main__":
    main()
