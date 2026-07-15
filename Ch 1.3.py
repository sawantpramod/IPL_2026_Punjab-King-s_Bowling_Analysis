"""
Research Question:
Did PBKS concede a significantly different death-over economy rate
between wins and losses?

H0:
There is no significant difference in PBKS's death-over economy
between wins and losses.

H1:
There is a significant difference in PBKS's death-over economy
between wins and losses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -------------------------------------------------------
# PBKS Bowling
# -------------------------------------------------------
pbks = df[df["bowling_team"] == "PBKS"].copy()

# -------------------------------------------------------
# Death Overs (16-20)
# If overs are 0-19, use >=15
# -------------------------------------------------------
death = pbks[pbks["over_no"] >= 15].copy()

# -------------------------------------------------------
# Economy per innings
# -------------------------------------------------------
death_summary = (
    death
    .groupby(
        ["match_id","innings_no","bowling_result"],
        as_index=False
    )
    .agg(
        runs_conceded=("total_runs","sum"),
        legal_balls=("is_legal_delivery","sum")
    )
)

# -------------------------------------------------------
# Overs Bowled
# -------------------------------------------------------
death_summary["overs_bowled"] = (
    death_summary["legal_balls"] / 6
)

# avoid divide-by-zero
death_summary = death_summary[
    death_summary["overs_bowled"] > 0
].copy()

# -------------------------------------------------------
# Economy
# -------------------------------------------------------
death_summary["death_economy"] = (
    death_summary["runs_conceded"] /
    death_summary["overs_bowled"]
)

print(death_summary.head())

# -------------------------------------------------------
# Wins / Losses
# -------------------------------------------------------
wins = death_summary.loc[
    death_summary["bowling_result"]=="Win",
    "death_economy"
]

losses = death_summary.loc[
    death_summary["bowling_result"]=="Loss",
    "death_economy"
]

print("\nWins")
print(wins.describe())

print("\nLosses")
print(losses.describe())

# -------------------------------------------------------
# Welch t-test
# -------------------------------------------------------
t_stat,p_value = ttest_ind(
    wins,
    losses,
    equal_var=False,
    nan_policy="omit"
)

print("\nWelch's t-test")
print("------------------------")
print(f"T-statistic : {t_stat:.3f}")
print(f"P-value     : {p_value:.5f}")

# -------------------------------------------------------
# Cohen's d
# -------------------------------------------------------
def cohens_d(x,y):

    x=np.asarray(x)
    y=np.asarray(y)

    pooled=np.sqrt(
        (
            ((len(x)-1)*np.var(x,ddof=1))
            +
            ((len(y)-1)*np.var(y,ddof=1))
        )/(len(x)+len(y)-2)
    )

    if pooled==0:
        return 0

    return (x.mean()-y.mean())/pooled

d=cohens_d(wins,losses)

abs_d=abs(d)

if abs_d<0.20:
    effect="Negligible"
elif abs_d<0.50:
    effect="Small"
elif abs_d<0.80:
    effect="Medium"
else:
    effect="Large"

print(f"\nCohen's d : {d:.3f}")
print("Effect Size :",effect)

# -------------------------------------------------------
# Average Economy
# -------------------------------------------------------
avg=death_summary.groupby(
    "bowling_result"
)["death_economy"].mean()

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
plt.figure(figsize=(6,5))

bars=plt.bar(avg.index,avg.values)

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.10,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("PBKS Death Over Economy")
plt.xlabel("Match Result")
plt.ylabel("Economy Rate")

plt.grid(axis="y",alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Final Verdict
# -------------------------------------------------------
mean_win=wins.mean()
mean_loss=losses.mean()

print("\n================ FINAL VERDICT ================\n")

print(f"Average Economy (Wins)   : {mean_win:.2f}")
print(f"Average Economy (Losses) : {mean_loss:.2f}")

print(f"\nP-value  : {p_value:.5f}")
print(f"Cohen's d: {d:.3f} ({effect})")

if p_value < 0.05:

    print("\nDecision : Reject H₀")

    if mean_loss > mean_win:

        print(
            "PBKS conceded a significantly higher death-over economy "
            "in losses than in wins."
        )

    else:

        print(
            "PBKS unexpectedly had a significantly higher death-over "
            "economy in victories."
        )

else:

    print("\nDecision : Fail to Reject H₀")

    if mean_loss > mean_win:

        print(
            "Although PBKS's death-over economy was higher in losses, "
            "the difference is not statistically significant."
        )

    else:

        print(
            "There is no statistically significant difference in "
            "death-over economy between wins and losses."
        )