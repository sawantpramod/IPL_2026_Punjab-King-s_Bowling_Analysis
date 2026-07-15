"""
Objective:
Did PBKS take significantly fewer Powerplay bowling wickets
in matches they lost?

H0:
There is no significant difference in Powerplay bowling wickets
between wins and losses.

H1:
PBKS took fewer Powerplay bowling wickets in losses than wins.

Run-outs are excluded.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -------------------------------------------------------
# Filter PBKS Bowling
# -------------------------------------------------------
pbks = df[df["bowling_team"] == "PBKS"].copy()

# -------------------------------------------------------
# Remove Run Outs
# -------------------------------------------------------
pbks = pbks[
    ~pbks["wicket_type"]
      .fillna("")
      .str.lower()
      .str.contains("run out")
].copy()

# -------------------------------------------------------
# Powerplay Overs (0-5)
# -------------------------------------------------------
powerplay = pbks[pbks["over_no"] < 6].copy()

# -------------------------------------------------------
# Count Bowling Wickets
# wicket_flag is boolean
# True = 1
# False = 0
# -------------------------------------------------------
pp_wickets = (
    powerplay
    .groupby(
        ["match_id", "innings_no", "bowling_result"],
        as_index=False
    )
    .agg(
        powerplay_wickets=("wicket_flag", "sum")
    )
)

print(pp_wickets.head())

# -------------------------------------------------------
# Split Wins and Losses
# -------------------------------------------------------
wins = pp_wickets.loc[
    pp_wickets["bowling_result"]=="Win",
    "powerplay_wickets"
]

losses = pp_wickets.loc[
    pp_wickets["bowling_result"]=="Loss",
    "powerplay_wickets"
]

# -------------------------------------------------------
# Descriptive Statistics
# -------------------------------------------------------
print("\nWins")
print(wins.describe())

print("\nLosses")
print(losses.describe())

# -------------------------------------------------------
# Welch's t-test
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

    nx=len(x)
    ny=len(y)

    pooled_sd=np.sqrt(
        (
            (nx-1)*np.var(x,ddof=1)
            +
            (ny-1)*np.var(y,ddof=1)
        )/(nx+ny-2)
    )

    return (np.mean(x)-np.mean(y))/pooled_sd

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
# Plot
# -------------------------------------------------------
avg=(
    pp_wickets
    .groupby("bowling_result")["powerplay_wickets"]
    .mean()
)

plt.figure(figsize=(6,5))

bars=plt.bar(avg.index,avg.values)

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.05,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("PBKS Average Powerplay Bowling Wickets")
plt.xlabel("Match Result")
plt.ylabel("Average Wickets")

plt.grid(axis="y",alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Final Verdict
# -------------------------------------------------------
mean_win=wins.mean()
mean_loss=losses.mean()

print("\n================ FINAL VERDICT ================\n")

print(f"Average PP Wickets (Wins)   : {mean_win:.2f}")
print(f"Average PP Wickets (Losses) : {mean_loss:.2f}")

print(f"\nP-value  : {p_value:.5f}")
print(f"Cohen's d: {d:.3f} ({effect})")

if p_value < 0.05:

    print("\nDecision : Reject H₀")

    if mean_loss < mean_win:

        print(
            "PBKS took significantly fewer Powerplay bowling wickets "
            "in matches they lost."
        )

    else:

        print(
            "PBKS unexpectedly took significantly more Powerplay wickets "
            "in matches they lost."
        )

else:

    print("\nDecision : Fail to Reject H₀")

    print(
        "There is no statistically significant difference in "
        "Powerplay bowling wickets between wins and losses."
    )