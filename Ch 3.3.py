# middle order boundary% by pbks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------- CHANGE 1: Filter for PBKS -------------------
df = df[df['bowling_team'] == 'PBKS']   # ← use your team's exact name

# ------------------------------------
# Middle Overs
# ------------------------------------
phase_df = df[(df["over_no"] >= 6) & (df["over_no"] < 15)].copy()

# Boundary indicators
phase_df["fours"] = (phase_df["runs_off_bat"] == 4).astype(int)
phase_df["sixes"] = (phase_df["runs_off_bat"] == 6).astype(int)
phase_df["boundary"] = phase_df["runs_off_bat"].isin([4,6]).astype(int)

# Team-wise summary
summary = (
    phase_df.groupby("bowling_team")
    .agg(
        Balls=("ball_no","count"),
        Fours=("fours","sum"),
        Sixes=("sixes","sum"),
        Boundary_Balls=("boundary","sum")
    )
)

summary["Boundary %"] = (
    summary["Boundary_Balls"] /
    summary["Balls"] * 100
).round(2)

print(summary)

# Plot
ax = summary[["Fours","Sixes","Boundary %"]].plot(
    kind="bar",
    figsize=(14,6)
)

plt.title("Middle Overs Boundary Statistics")
plt.xlabel("Bowling Team")
plt.ylabel("Value")
plt.xticks(rotation=45)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.show()