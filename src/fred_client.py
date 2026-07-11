from __future__ import annotations

import os
from typing import Any

import pandas as pd


DEFAULT_MACRO_VALUES = {
    "unrate": 4.2,
    "fedfunds": 4.5,
    "cpi": 320.0,
}


def get_macro_indicators(api_key: str | None = None) -> dict[str, float | str]:
    """Fetch the latest macro values from FRED when available, otherwise fall back to defaults."""

    api_key = api_key or os.getenv("FRED_API_KEY")

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key) if api_key else Fred()
        unrate = float(fred.get_series("UNRATE").dropna().iloc[-1])
        fedfunds = float(fred.get_series("FEDFUNDS").dropna().iloc[-1])
        cpi = float(fred.get_series("CPIAUCSL").dropna().iloc[-1])
        return {"unrate": unrate, "fedfunds": fedfunds, "cpi": cpi, "_source": "fred"}
    except Exception:
        return {**dict(DEFAULT_MACRO_VALUES), "_source": "default"}


def get_macro_history(api_key: str | None = None, periods: int = 24) -> pd.DataFrame:
    """Return a small history frame for dashboard charts using FRED if possible."""

    api_key = api_key or os.getenv("FRED_API_KEY")

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key) if api_key else Fred()
        data = {
            "unrate": fred.get_series("UNRATE", observation_start="2019-01-01"),
            "fedfunds": fred.get_series("FEDFUNDS", observation_start="2019-01-01"),
            "cpi": fred.get_series("CPIAUCSL", observation_start="2019-01-01"),
        }
        frame = pd.DataFrame(data).dropna()
        return frame.tail(periods)
    except Exception:
        return pd.DataFrame([DEFAULT_MACRO_VALUES], index=[pd.Timestamp.today()])
