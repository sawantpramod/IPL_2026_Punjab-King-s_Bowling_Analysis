""" This is an excellent pressure metric. However, I'd recommend measuring Strike Rotation % as the percentage of legal balls that result in exactly one run (excluding wides/no-balls because they are not legal deliveries). This directly reflects how often batters could rotate the strike.

Research Question

Did PBKS allow significantly more strike rotation (singles) in matches they lost?

Hypotheses
H₀: There is no difference in strike rotation percentage between wins and losses.
H₁: PBKS allowed a higher strike rotation percentage in losses."""

# ============================================================
# Strike Rotation Analysis
# PBKS Bowling
# Metric: Singles % and Boundary %
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
    pbks.groupby(["match_id", "innings_no", "bowling_result"])
        .agg(
            legal_balls=("is_legal_delivery", "sum"),
            singles=("runs_off_bat", lambda x: (x == 1).sum()),
            boundaries=("runs_off_bat", lambda x: x.isin([4, 6]).sum())
        )
        .reset_index()
)

# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

summary["single_pct"] = (
    summary["singles"] /
    summary["legal_balls"] * 100
)

summary["boundary_pct"] = (
    summary["boundaries"] /
    summary["legal_balls"] * 100
)

print(summary.head())

# Function for Welch's t-test

def cohens_d(x, y):

    nx = len(x)
    ny = len(y)

    pooled_sd = np.sqrt(
        (
            (nx-1)*np.var(x, ddof=1) +
            (ny-1)*np.var(y, ddof=1)
        )/(nx+ny-2)
    )

    return (np.mean(x)-np.mean(y))/pooled_sd


def analyze(metric):

    win = summary.loc[
        summary["bowling_result"]=="Win",
        metric
    ]

    loss = summary.loc[
        summary["bowling_result"]=="Loss",
        metric
    ]

    t_stat, p_value = ttest_ind(
        loss,
        win,
        equal_var=False,
        alternative="greater"
    )

    d = cohens_d(loss, win)

    print("\n====================================")
    print(metric.upper())
    print("====================================")

    print(f"Win Mean  : {win.mean():.2f}%")
    print(f"Loss Mean : {loss.mean():.2f}%")

    print(f"T-statistic : {t_stat:.3f}")
    print(f"P-value     : {p_value:.6f}")

    print(f"Cohen's d   : {d:.3f}")

    # Interpretation
    alpha = 0.05

    if p_value < alpha:
        print("Interpretation: Reject H₀.")
        print(f"PBKS allowed a significantly higher {metric.replace('_',' ')} in losses.")
    else:
        print("Interpretation: Fail to reject H₀.")
        print(f"No significant difference in {metric.replace('_',' ')} between wins and losses.")

    # Effect Size
    if abs(d) < 0.20:
        effect = "Negligible"
    elif abs(d) < 0.50:
        effect = "Small"
    elif abs(d) < 0.80:
        effect = "Medium"
    else:
        effect = "Large"

    print(f"Effect Size : {effect}")

# run analysis

analyze("single_pct")

analyze("boundary_pct")