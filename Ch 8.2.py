""" Strike Rotation Index (SRI) rather than just Singles %. It is a more comprehensive metric because it captures all common forms of strike rotation (1s, 2s, and 3s).

Research Question

Did PBKS allow significantly more strike rotation in matches they lost?

Metric
Strike Rotation Index (SRI)=
Legal Balls
Singles + Twos + Threes
	​

×100
Hypotheses
H₀: There is no difference in Strike Rotation Index between wins and losses.
H₁: PBKS allowed a higher Strike Rotation Index in losses."""

# ============================================================
# Strike Rotation Index (SRI)
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
    pbks.groupby(["match_id", "innings_no", "bowling_result"])
        .agg(
            legal_balls=("is_legal_delivery", "sum"),
            singles=("runs_off_bat", lambda x: (x == 1).sum()),
            twos=("runs_off_bat", lambda x: (x == 2).sum()),
            threes=("runs_off_bat", lambda x: (x == 3).sum())
        )
        .reset_index()
)

# ------------------------------------------------------------
# Strike Rotation Balls
# ------------------------------------------------------------

summary["rotation_balls"] = (
    summary["singles"] +
    summary["twos"] +
    summary["threes"]
)

# ------------------------------------------------------------
# Strike Rotation Index (%)
# ------------------------------------------------------------

summary["strike_rotation_index"] = (
    summary["rotation_balls"] /
    summary["legal_balls"] * 100
)

print(summary.head())

# Welch's t-test & Cohen's d

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = summary.loc[
    summary["bowling_result"] == "Win",
    "strike_rotation_index"
]

loss = summary.loc[
    summary["bowling_result"] == "Loss",
    "strike_rotation_index"
]

# ------------------------------------------------------------
# Welch's t-test
# H1 : Loss > Win
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
            (nx-1) * np.var(x, ddof=1) +
            (ny-1) * np.var(y, ddof=1)
        ) / (nx + ny - 2)
    )

    return (np.mean(x) - np.mean(y)) / pooled_sd

d = cohens_d(loss, win)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n==============================")
print("Strike Rotation Index")
print("==============================")

print(f"Win Mean  : {win.mean():.2f}%")
print(f"Loss Mean : {loss.mean():.2f}%")

print(f"\nWelch's t-test")
print(f"T-statistic : {t_stat:.3f}")
print(f"P-value     : {p_value:.6f}")

print(f"\nCohen's d : {d:.3f}")

# Interpretation

alpha = 0.05

print("\nInterpretation")

if p_value < alpha:
    print("Reject H₀.")
    print("PBKS allowed significantly more strike rotation in losses.")
else:
    print("Fail to reject H₀.")
    print("No significant difference in strike rotation between wins and losses.")

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