from __future__ import annotations

from typing import Any


def score_probability(probability_of_default: float) -> dict[str, Any]:
    """Convert a PD into score, risk band, and business decision."""

    if not 0.0 <= probability_of_default <= 1.0:
        raise ValueError("Probability of default must be between 0.0 and 1.0")

    score = int(round(850 - (probability_of_default * 550)))

    if probability_of_default < 0.10:
        risk_band = "Low"
        decision = "Approve"
    elif probability_of_default < 0.25:
        risk_band = "Medium"
        decision = "Manual Review"
    else:
        risk_band = "High"
        decision = "Reject"

    return {
        "probability_of_default": probability_of_default,
        "credit_score": score,
        "risk_band": risk_band,
        "decision": decision,
    }
