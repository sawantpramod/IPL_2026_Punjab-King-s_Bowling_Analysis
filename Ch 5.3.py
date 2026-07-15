# Middle order wise analysis of bowling pressure index 
"""
Middle order Bowling Pressure Index (PBKS)

Pressure Index = Dot Ball % + Wicket % − Boundary %

Dot Ball % = (Dot Balls / Legal Balls) × 100
Wicket % = (Wickets / Legal Balls) × 100
Boundary % = (Fours + Sixes) / Legal Balls × 100
"""

# ============================================================
# middle over Bowling Pressure Index (PBKS)
# Overs 6-14
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ------------------------------------------------------------
# Filter PBKS Bowling - Middle Overs (Overs 6-14)
# ------------------------------------------------------------

middle = df[
    (df["bowling_team"] == "PBKS") &
    (df["over_no"] >= 6) &
    (df["over_no"] <= 14)
].copy()

# ------------------------------------------------------------
# Dot Ball Indicator
# ------------------------------------------------------------

middle["dot_ball"] = (
    (middle["is_legal_delivery"]) &
    (middle["total_runs"] == 0)
)

# ------------------------------------------------------------
# Match-wise Summary
# ------------------------------------------------------------

summary = (
    middle.groupby(["match_id", "innings_no", "bowling_result"])
      .agg(
          legal_balls=("is_legal_delivery", "sum"),
          dot_balls=("dot_ball", "sum"),
          wickets=("wicket_flag", "sum"),
          boundaries=("runs_off_bat",
                      lambda x: x.isin([4, 6]).sum())
      )
      .reset_index()
)

# ------------------------------------------------------------
# Remove innings with no legal deliveries (safety check)
# ------------------------------------------------------------

summary = summary[summary["legal_balls"] > 0]

# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

summary["dot_pct"] = (
    summary["dot_balls"] /
    summary["legal_balls"] * 100
)

summary["wicket_pct"] = (
    summary["wickets"] /
    summary["legal_balls"] * 100
)

summary["boundary_pct"] = (
    summary["boundaries"] /
    summary["legal_balls"] * 100
)

# ------------------------------------------------------------
# Bowling Pressure Index
# ------------------------------------------------------------

summary["pressure_index"] = (
    summary["dot_pct"]
    + summary["wicket_pct"]
    - summary["boundary_pct"]
)

print(summary.head())

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = summary.loc[
    summary["bowling_result"] == "Win",
    "pressure_index"
]

loss = summary.loc[
    summary["bowling_result"] == "Loss",
    "pressure_index"
]

# ------------------------------------------------------------
# Check sample size
# ------------------------------------------------------------

print("\nSample Size")
print(f"Win  : {len(win)}")
print(f"Loss : {len(loss)}")

if len(win) < 2 or len(loss) < 2:

    print("\nNot enough observations to perform Welch's t-test.")

else:

    # --------------------------------------------------------
    # Welch's t-test
    # H1 : Win > Loss
    # --------------------------------------------------------

    t_stat, p_value = ttest_ind(
        win,
        loss,
        equal_var=False,
        alternative="greater"
    )

    # --------------------------------------------------------
    # Cohen's d
    # --------------------------------------------------------

    def cohens_d(x, y):

        nx = len(x)
        ny = len(y)

        pooled = np.sqrt(
            (
                (nx-1) * np.var(x, ddof=1)
                +
                (ny-1) * np.var(y, ddof=1)
            ) / (nx + ny - 2)
        )

        return (np.mean(x) - np.mean(y)) / pooled

    d = cohens_d(win, loss)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n==============================")
    print("Middleover Bowling Pressure Index")
    print("==============================")

    print(f"Win Mean  : {win.mean():.2f}")
    print(f"Loss Mean : {loss.mean():.2f}")

    print("\nWelch's t-test")
    print(f"T-statistic : {t_stat:.3f}")
    print(f"P-value     : {p_value:.6f}")

    print(f"\nCohen's d : {d:.3f}")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    alpha = 0.05

    print("\nInterpretation")

    if p_value < alpha:
        print("Reject H₀.")
        print("PBKS had a significantly higher Middle Over Bowling Pressure Index in wins.")
    else:
        print("Fail to reject H₀.")
        print("No significant difference in Middle Over Bowling Pressure Index between wins and losses.")

    # Effect Size

    if abs(d) < 0.20:
        effect = "Negligible"
    elif abs(d) < 0.50:
        effect = "Small"
    elif abs(d) < 0.80:
        effect = "Medium"
    else:
        effect = "Large"

    print(f"Cohen's d = {d:.2f} ({effect} effect)")