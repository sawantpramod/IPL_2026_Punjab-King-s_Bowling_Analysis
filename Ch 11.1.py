# ============================================================
# Point-Biserial Correlation
# Economy Rate vs Winning
# Phase-wise PBKS Bowling Analysis
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

# ------------------------------------------------------------
# Filter PBKS Bowling
# ------------------------------------------------------------

pbks = df[df["bowling_team"] == "PBKS"].copy()

# ------------------------------------------------------------
# Create Phase
# ------------------------------------------------------------

conditions = [
    pbks["over_no"].between(0, 5),
    pbks["over_no"].between(6, 14),
    pbks["over_no"].between(15, 19)
]

choices = [
    "Powerplay",
    "Middle Overs",
    "Death Overs"
]

pbks["phase"] = np.select(
    conditions,
    choices,
    default="Other"
)

# ------------------------------------------------------------
# Phase-wise Economy Per Match
# ------------------------------------------------------------

phase_summary = (
    pbks.groupby(
        ["match_id", "innings_no", "phase", "bowling_result"],
        as_index=False
    )
    .agg(
        runs=("total_runs", "sum"),
        legal_balls=("is_legal_delivery", "sum")
    )
)

# ------------------------------------------------------------
# Economy
# ------------------------------------------------------------

phase_summary = phase_summary[
    phase_summary["legal_balls"] > 0
]

phase_summary["economy"] = (
    phase_summary["runs"] /
    (phase_summary["legal_balls"] / 6)
)

# ------------------------------------------------------------
# Encode Win/Loss
# ------------------------------------------------------------

phase_summary["win"] = np.where(
    phase_summary["bowling_result"] == "Win",
    1,
    0
)

# ------------------------------------------------------------
# Correlation Phase-wise
# ------------------------------------------------------------

for phase in ["Powerplay", "Middle Overs", "Death Overs"]:

    temp = phase_summary[
        phase_summary["phase"] == phase
    ].copy()

    print("\n" + "="*50)
    print(phase)
    print("="*50)

    print(f"Observations : {len(temp)}")

    # Need at least 3 rows and both win/loss groups
    if len(temp) < 3:
        print("Not enough observations.")
        continue

    if temp["win"].nunique() < 2:
        print("Only one match result present.")
        continue

    r, p = pointbiserialr(
        temp["win"],
        temp["economy"]
    )

    print(f"Correlation : {r:.3f}")
    print(f"P-value     : {p:.4f}")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("\nInterpretation:")

    if p < 0.05:

        if r < 0:
            print(
                "Lower economy rates were significantly associated "
                "with a higher probability of winning."
            )

        else:
            print(
                "Higher economy rates were significantly associated "
                "with a higher probability of winning."
            )

    else:

        print(
            "No statistically significant relationship between "
            "economy rate and winning."
        )

    # Strength
    abs_r = abs(r)

    if abs_r < 0.10:
        strength = "Negligible"
    elif abs_r < 0.30:
        strength = "Weak"
    elif abs_r < 0.50:
        strength = "Moderate"
    else:
        strength = "Strong"

    print(f"Relationship Strength: {strength}")