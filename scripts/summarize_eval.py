"""Summarize the DR-study evaluation matrix (policy x world).

Reads ~/agile/logs/dr_eval/<policy>/<world>/trajectories/ written by run_eval_matrix.sh and prints:
  - success rate (episodes that reached the end without falling)
  - forward-velocity tracking RMSE in the walking phases
  - pelvis-height tracking RMSE in the squat / stand-up phases
Also writes ~/agile/logs/dr_eval/summary.csv

Usage:
    ~/agile/run_agile.sh scripts/summarize_eval.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="~/agile/logs/dr_eval")
parser.add_argument("--policy_dt", type=float, default=0.02)
parser.add_argument("--show_phases", action="store_true", help="print the mean command per phase (sanity check)")
args = parser.parse_args()

# Phases of eval_squat_walk.yaml. The first 0.5 s after each command change is skipped (transition).
PHASES = {
    "squat":    (3.5, 6.0),
    "stand_up": (6.5, 9.0),
    "walk_1.0": (9.5, 19.0),
    "walk_1.5": (19.5, 24.0),
    "turn":     (24.5, 29.0),
}
HEIGHT_PHASES = ["squat", "stand_up"]
WALK_PHASES = ["walk_1.0", "walk_1.5", "turn"]

POLICIES = ["nominal", "a", "b"]
WORLDS = ["NoTS", "Sat2p0", "Sat1p5", "Sat1p0", "AGILE"]

root = Path(args.root).expanduser()


def load(policy, world):
    d = root / policy / world / "trajectories"
    files = sorted(d.glob("episode_*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f).assign(episode=i) for i, f in enumerate(files)], ignore_index=True)
    df["t"] = df["frame_idx"] * args.policy_dt
    return df


def phase_rmse(df, phase, err_col):
    t0, t1 = PHASES[phase]
    sel = df[(df["t"] >= t0) & (df["t"] < t1)]
    return float(np.sqrt(np.mean(sel[err_col] ** 2))) if len(sel) else np.nan


rows = []
shown = False
for p in POLICIES:
    for w in WORLDS:
        df = load(p, w)
        if df is None:
            continue
        # commands = [vx, vy, wz, height]
        df["err_vx"] = df["commands_0"] - df["root_lin_vel_robot_0"]
        df["err_h"] = df["commands_3"] - df["root_pos_2"]
        if args.show_phases and not shown:
            print(f"\nMean command per phase ({p}/{w}) - check that the columns are [vx, vy, wz, h]:")
            for name, (t0, t1) in PHASES.items():
                sel = df[(df["t"] >= t0) & (df["t"] < t1)]
                print(f"  {name:9s}", np.round(sel[[f'commands_{i}' for i in range(4)]].mean().to_numpy(), 2))
            shown = True
        succ = df.groupby("episode")["is_success"].first().mean()
        row = {"policy": p, "world": w, "episodes": df["episode"].nunique(), "success": succ}
        for ph in HEIGHT_PHASES:
            row[f"h_rmse_{ph}"] = phase_rmse(df, ph, "err_h")
        for ph in WALK_PHASES:
            row[f"vx_rmse_{ph}"] = phase_rmse(df, ph, "err_vx")
        rows.append(row)

if not rows:
    raise SystemExit(f"No results under {root}. Run run_eval_matrix.sh first.")

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
out = root / "summary.csv"
table.to_csv(out, index=False)
print(f"\nSaved {out}")
print("success = fraction of episodes without a fall | h_rmse in m | vx_rmse in m/s")
