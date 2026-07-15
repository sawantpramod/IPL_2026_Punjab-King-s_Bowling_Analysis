# Pressure Analysis
# ============================================================
# Objective:
# Did PBKS bowl significantly fewer dot balls in the death over
# during matches they lost?
#
# H0:
# There is no difference in death over dot-ball percentage
# between wins and losses.
#
# H1:
# PBKS bowled a lower dot-ball percentage in losses.
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ------------------------------------------------------------
# Filter PBKS Bowling - deathover
# ------------------------------------------------------------

deathover = df[
    (df["bowling_team"] == "PBKS") &
    (df["over_no"] >= 15)
].copy()

# ------------------------------------------------------------
# Dot Ball Definition
# Legal delivery with zero total runs
# ------------------------------------------------------------

deathover["dot_ball"] = (
    (deathover["is_legal_delivery"] == True) &
    (deathover["total_runs"] == 0)
)

# ------------------------------------------------------------
# Match-wise Dot Ball %
# ------------------------------------------------------------

dot_ball_pct = (
    deathover.groupby(["match_id", "innings_no", "bowling_result"])
      .agg(
          legal_balls=("is_legal_delivery", "sum"),
          dot_balls=("dot_ball", "sum")
      )
      .reset_index()
)

dot_ball_pct["dot_ball_pct"] = (
    dot_ball_pct["dot_balls"] /
    dot_ball_pct["legal_balls"] * 100
)

print(dot_ball_pct.head())

# ------------------------------------------------------------
# Split Win vs Loss
# ------------------------------------------------------------

win = dot_ball_pct.loc[
    dot_ball_pct["bowling_result"]=="Win",
    "dot_ball_pct"
]

loss = dot_ball_pct.loc[
    dot_ball_pct["bowling_result"]=="Loss",
    "dot_ball_pct"
]

# ------------------------------------------------------------
# Welch's t-test
# Alternative:
# Win > Loss
# ------------------------------------------------------------

t_stat, p_value = ttest_ind(
    win,
    loss,
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
            (nx-1)*np.var(x, ddof=1) +
            (ny-1)*np.var(y, ddof=1)
        ) /
        (nx+ny-2)
    )

    return (np.mean(x)-np.mean(y))/pooled_sd

d = cohens_d(win, loss)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n==============================")
print("Deathover Dot Ball %")
print("==============================")

print(f"Win Mean  : {win.mean():.2f}%")
print(f"Loss Mean : {loss.mean():.2f}%")

print(f"\nWelch's t-test")
print(f"T-statistic : {t_stat:.4f}")
print(f"P-value     : {p_value:.6f}")

print(f"\nCohen's d : {d:.3f}")

# Interpretation
alpha = 0.05

print("\nInterpretation")

if p_value < alpha:
    print("Reject H₀.")
    print("PBKS bowled significantly fewer dot balls in deathover during losses.")
else:
    print("Fail to reject H₀.")
    print("No significant difference in deathover dot-ball percentage between wins and losses.")

# Cohen's d
if abs(d) < 0.2:
    effect = "Negligible"
elif abs(d) < 0.5:
    effect = "Small"
elif abs(d) < 0.8:
    effect = "Medium"
else:
    effect = "Large"

print(f"Cohen's d = {d:.2f} ({effect} effect)")