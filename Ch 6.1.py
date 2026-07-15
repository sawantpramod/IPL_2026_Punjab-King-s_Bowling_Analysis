""" After PBKS took a wicket, how many runs did they concede in the next 12 legal deliveries?

A lower value means the bowlers maintained pressure after the wicket. A higher value suggests the batting team regained momentum quickly.

Research Question

Did PBKS concede significantly more runs after taking a wicket in matches they lost?

Hypotheses
H₀: There is no difference in runs conceded after a wicket between wins and losses.
H₁: PBKS conceded more runs after taking a wicket in losses."""

# ============================================================
# Bowling Momentum
# Runs Conceded in Next 12 Legal Balls After a Wicket
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
# Sort Ball-by-Ball
# ------------------------------------------------------------

pbks = pbks.sort_values(
    ["match_id", "innings_no", "over_no", "ball_no"]
).reset_index(drop=True)

# ------------------------------------------------------------
# Runs after every wicket
# ------------------------------------------------------------

records = []

for (match_id, innings_no), innings in pbks.groupby(
        ["match_id", "innings_no"]):

    innings = innings.reset_index(drop=True)

    wicket_idx = innings.index[innings["wicket_flag"] == True]

    for idx in wicket_idx:

        # Next 12 legal deliveries
        future = innings.iloc[idx+1:].copy()

        future = future[future["is_legal_delivery"]]

        future = future.head(12)

        records.append({
            "match_id": match_id,
            "innings_no": innings_no,
            "bowling_result":
                innings.loc[idx, "bowling_result"],
            "runs_after_wicket":
                future["total_runs"].sum()
        })

momentum = pd.DataFrame(records)

print(momentum.head())

# Welch's t-test + Cohen's d

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = momentum.loc[
    momentum["bowling_result"] == "Win",
    "runs_after_wicket"
]

loss = momentum.loc[
    momentum["bowling_result"] == "Loss",
    "runs_after_wicket"
]

print("\nSample Size")
print("Win :", len(win))
print("Loss:", len(loss))

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

    pooled = np.sqrt(
        (
            (nx-1) * np.var(x, ddof=1) +
            (ny-1) * np.var(y, ddof=1)
        ) /
        (nx + ny - 2)
    )

    return (np.mean(x) - np.mean(y)) / pooled

d = cohens_d(loss, win)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n==============================")
print("Bowling Momentum Analysis")
print("==============================")

print(f"Win Mean  : {win.mean():.2f} runs")
print(f"Loss Mean : {loss.mean():.2f} runs")

print(f"\nT-statistic : {t_stat:.3f}")
print(f"P-value     : {p_value:.6f}")

print(f"Cohen's d   : {d:.3f}")

# Interpretation

alpha = 0.05

print("\nInterpretation")

if p_value < alpha:
    print("Reject H₀.")
    print("PBKS conceded significantly more runs after taking wickets in losses.")
else:
    print("Fail to reject H₀.")
    print("No significant difference in runs conceded after taking wickets.")

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