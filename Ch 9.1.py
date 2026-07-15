# Hypotheses
# H₀: There is no significant difference in extras conceded between wins and losses.
# H₁: PBKS conceded more extras in losses.
# ============================================================
# Objective:
# Did PBKS concede significantly more extras in matches they lost?
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ------------------------------------------------------------
# Filter PBKS Bowling
# ------------------------------------------------------------

pbks = df[df["bowling_team"] == "PBKS"].copy()

# ------------------------------------------------------------
# Extras per innings
# ------------------------------------------------------------

extras_df = (
    pbks.groupby(["match_id", "innings_no", "bowling_result"])
        .agg(
            extras=("extras", "sum")
        )
        .reset_index()
)

print(extras_df.head())

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = extras_df.loc[
    extras_df["bowling_result"] == "Win",
    "extras"
]

loss = extras_df.loc[
    extras_df["bowling_result"] == "Loss",
    "extras"
]

# ------------------------------------------------------------
# Welch's t-test
# H1: Loss > Win
# ------------------------------------------------------------

t_stat, p_value = ttest_ind(
    loss,
    win,
    equal_var=False,
    alternative="greater"
)

# ------------------------------------------------------------
# Cohen's d
# ------------------------------------------------------------

def cohens_d(x, y):

    nx = len(x)
    ny = len(y)

    pooled_sd = np.sqrt(
        (
            (nx - 1) * np.var(x, ddof=1) +
            (ny - 1) * np.var(y, ddof=1)
        ) / (nx + ny - 2)
    )

    return (np.mean(x) - np.mean(y)) / pooled_sd

d = cohens_d(loss, win)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n==============================")
print("Extras Analysis")
print("==============================")

print(f"Mean Extras (Win)  : {win.mean():.2f}")
print(f"Mean Extras (Loss) : {loss.mean():.2f}")

print(f"\nWelch's t-test")
print(f"T-statistic : {t_stat:.4f}")
print(f"P-value     : {p_value:.6f}")

print(f"\nCohen's d : {d:.3f}")

# interpretation

alpha = 0.05

print("\nInterpretation")

if p_value < alpha:
    print("Reject H₀.")
    print("PBKS conceded significantly more extras in losses.")
else:
    print("Fail to reject H₀.")
    print("No significant difference in extras conceded between wins and losses.")

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