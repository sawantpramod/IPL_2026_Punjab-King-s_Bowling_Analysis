""" Clutch Index=
Death Overs
Dot Balls+2×Wickets−Boundaries ​
is good, but it depends on the number of overs actually bowled. A more comparable metric across innings is to normalize by legal balls instead of overs.

Recommended Death Over Clutch Index (DOCI)
DOCI=
Legal Balls
Dot Balls+2×Wickets−Boundaries ​ ×100 ​ This has several advantages:
Accounts for shortened innings.
Comparable across matches.
Produces an intuitive score (higher = better clutch bowling)."""

# ============================================================
# Death Over Clutch Index (DOCI)
# PBKS Bowling
# Death Overs (15-19)
# Only while defending (2nd innings)
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ------------------------------------------------------------
# Filter PBKS Bowling
# Death Overs while defending
# ------------------------------------------------------------

death = df[
    (df["bowling_team"] == "PBKS") &
    (df["innings_no"] == 2) &
    (df["over_no"] >= 15) &
    (df["over_no"] <= 19)
].copy()

# ------------------------------------------------------------
# Dot Ball Indicator
# ------------------------------------------------------------

death["dot_ball"] = (
    (death["is_legal_delivery"]) &
    (death["total_runs"] == 0)
)

# ------------------------------------------------------------
# Match-wise Summary
# ------------------------------------------------------------

summary = (
    death.groupby(
        ["match_id", "innings_no", "bowling_result"],
        as_index=False
    )
    .agg(
        legal_balls=("is_legal_delivery","sum"),
        dot_balls=("dot_ball","sum"),
        wickets=("wicket_flag","sum"),
        boundaries=("runs_off_bat",
                    lambda x: x.isin([4,6]).sum()),
        runs=("total_runs","sum")
    )
)

summary = summary[summary["legal_balls"] > 0]

# ------------------------------------------------------------
# Death Over Clutch Index
# ------------------------------------------------------------

summary["clutch_index"] = (
    (
        summary["dot_balls"]
        + 2*summary["wickets"]
        - summary["boundaries"]
    )
    /
    summary["legal_balls"]
) * 100

print(summary.head())

# Welch's t-test

# ------------------------------------------------------------
# Win vs Loss
# ------------------------------------------------------------

win = summary.loc[
    summary["bowling_result"]=="Win",
    "clutch_index"
]

loss = summary.loc[
    summary["bowling_result"]=="Loss",
    "clutch_index"
]

print("\nSample Size")
print("Win :", len(win))
print("Loss:", len(loss))

if len(win) >= 2 and len(loss) >= 2:

    t_stat, p_value = ttest_ind(
        win,
        loss,
        equal_var=False,
        alternative="greater"
    )

    # Cohen's d

    def cohens_d(x,y):

        nx=len(x)
        ny=len(y)

        pooled=np.sqrt(
            (
                (nx-1)*np.var(x,ddof=1)+
                (ny-1)*np.var(y,ddof=1)
            )/(nx+ny-2)
        )

        return (np.mean(x)-np.mean(y))/pooled

    d=cohens_d(win,loss)

    print("\n==============================")
    print("Death Over Clutch Index")
    print("==============================")

    print(f"Win Mean  : {win.mean():.2f}")
    print(f"Loss Mean : {loss.mean():.2f}")

    print("\nWelch's t-test")
    print(f"T-statistic : {t_stat:.3f}")
    print(f"P-value     : {p_value:.6f}")

    print(f"\nCohen's d : {d:.3f}")

# interpretation

alpha = 0.05

print("\nInterpretation")

if p_value < alpha:

    print("Reject H₀.")
    print("PBKS showed significantly better clutch bowling in the death overs during wins.")

else:

    print("Fail to reject H₀.")
    print("No significant difference in death-over clutch bowling between wins and losses.")

# Effect size

if abs(d) < 0.20:
    effect="Negligible"
elif abs(d) < 0.50:
    effect="Small"
elif abs(d) < 0.80:
    effect="Medium"
else:
    effect="Large"

print(f"Cohen's d = {d:.2f} ({effect} effect)")