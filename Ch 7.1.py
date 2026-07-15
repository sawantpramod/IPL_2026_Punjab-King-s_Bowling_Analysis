""" Research Question

Did PBKS take fewer wickets per boundary conceded in matches they lost?

Metric
Wickets per Boundary= Boundaries Conceded
Wickets ​​
where:
Wickets = Total wickets taken
Boundaries = Fours + Sixes conceded
Hypotheses
H₀: There is no difference in Wickets per Boundary between wins and losses.
H₁: PBKS had a lower Wickets per Boundary ratio in losses."""

# ============================================================
# Wickets per Boundary Conceded
# PBKS Bowling Analysis
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ------------------------------------------------------------
# Filter PBKS Bowling
# ------------------------------------------------------------

pbks = df[df["bowling_team"] == "PBKS"].copy()

# ------------------------------------------------------------
# Match-wise Summary
# ------------------------------------------------------------

summary = (
    pbks.groupby(
        ["match_id", "innings_no", "bowling_result"],
        as_index=False
    )
    .agg(
        wickets=("wicket_flag", "sum"),
        boundaries=("runs_off_bat",
                    lambda x: x.isin([4, 6]).sum())
    )
)

# ------------------------------------------------------------
# Avoid division by zero
# ------------------------------------------------------------

summary["boundaries"] = summary["boundaries"].replace(0, np.nan)

summary["wickets_per_boundary"] = (
    summary["wickets"] /
    summary["boundaries"]
)

summary = summary.dropna(subset=["wickets_per_boundary"])

print(summary.head())

# Welch's t-test & Cohen's d

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = summary.loc[
    summary["bowling_result"]=="Win",
    "wickets_per_boundary"
]

loss = summary.loc[
    summary["bowling_result"]=="Loss",
    "wickets_per_boundary"
]

print("\nSample Size")
print("Win :", len(win))
print("Loss:", len(loss))

if len(win) >= 2 and len(loss) >= 2:

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
                (nx-1) * np.var(x, ddof=1) +
                (ny-1) * np.var(y, ddof=1)
            ) / (nx + ny - 2)
        )

        return (np.mean(x) - np.mean(y)) / pooled

    d = cohens_d(win, loss)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n==============================")
    print("Wickets per Boundary Conceded")
    print("==============================")

    print(f"Win Mean  : {win.mean():.3f}")
    print(f"Loss Mean : {loss.mean():.3f}")

    print(f"\nWelch's t-test")
    print(f"T-statistic : {t_stat:.3f}")
    print(f"P-value     : {p_value:.6f}")

    print(f"\nCohen's d : {d:.3f}")

else:

    print("Not enough observations for statistical testing.")

# Interpretation

alpha = 0.05

if len(win) >= 2 and len(loss) >= 2:

    print("\nInterpretation")

    if p_value < alpha:
        print("Reject H₀.")
        print("PBKS had a significantly higher Wickets per Boundary ratio in wins.")
    else:
        print("Fail to reject H₀.")
        print("No significant difference in Wickets per Boundary ratio between wins and losses.")

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