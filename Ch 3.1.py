"""Research Questions
Did PBKS concede a significantly higher Four % in matches they lost?
Did PBKS concede a significantly higher Six % in matches they lost?
Hypotheses (for both)
H₀: There is no difference in boundary percentage between wins and losses.
H₁: PBKS conceded a higher boundary percentage in losses."""

# ============================================================
# Boundary Type Analysis (4s and 6s)
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

boundary_df = (
    pbks.groupby(["match_id", "innings_no", "bowling_result"])
        .agg(
            legal_balls=("is_legal_delivery", "sum"),
            fours=("runs_off_bat", lambda x: (x == 4).sum()),
            sixes=("runs_off_bat", lambda x: (x == 6).sum())
        )
        .reset_index()
)

# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

boundary_df["four_pct"] = (
    boundary_df["fours"] /
    boundary_df["legal_balls"] * 100
)

boundary_df["six_pct"] = (
    boundary_df["sixes"] /
    boundary_df["legal_balls"] * 100
)

print(boundary_df.head())

# Welch's t-test + Cohen's d

# ------------------------------------------------------------
# Function
# ------------------------------------------------------------

def cohens_d(x, y):

    nx = len(x)
    ny = len(y)

    pooled = np.sqrt(
        (
            (nx-1)*np.var(x, ddof=1) +
            (ny-1)*np.var(y, ddof=1)
        )/(nx+ny-2)
    )

    return (np.mean(x)-np.mean(y))/pooled


def analyze(metric_name):

    win = boundary_df.loc[
        boundary_df["bowling_result"]=="Win",
        metric_name
    ]

    loss = boundary_df.loc[
        boundary_df["bowling_result"]=="Loss",
        metric_name
    ]

    t_stat, p_value = ttest_ind(
        loss,
        win,
        equal_var=False,
        alternative="greater"
    )

    d = cohens_d(loss, win)

    print("\n================================")
    print(metric_name.upper())
    print("================================")

    print(f"Win Mean  : {win.mean():.2f}%")
    print(f"Loss Mean : {loss.mean():.2f}%")

    print(f"T-statistic : {t_stat:.3f}")
    print(f"P-value     : {p_value:.6f}")

    print(f"Cohen's d   : {d:.3f}")

    # ----------------------------
    # Interpretation
    # ----------------------------

    alpha = 0.05

    if p_value < alpha:
        print("Interpretation: Reject H₀.")
        print(f"PBKS conceded a significantly higher {metric_name.replace('_',' ')} in losses.")
    else:
        print("Interpretation: Fail to reject H₀.")
        print(f"No significant difference in {metric_name.replace('_',' ')} between wins and losses.")

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

# ------------------------------------------------------------
# Run Analysis
# ------------------------------------------------------------

analyze("four_pct")

analyze("six_pct")