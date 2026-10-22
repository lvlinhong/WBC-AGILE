"""Summarize the DR-study evaluation matrix (policy x world x payload).

Reads ~/agile/logs/dr_eval_v2/<policy>/<world>_P<kg>/trajectories/ and prints per run:
  success        fraction of episodes without a fall
  h_rmse_*       pelvis-height tracking RMSE [m] in squat / stand-up
  vx_rmse_*      forward-velocity tracking RMSE [m/s] in walking phases
  clip_%         share of lower-body joint samples where the motor limit cut the torque
                 (computed - applied > 1 Nm), t >= 1 s
  knee_util_p95  95th percentile of |knee torque| / knee tau_lim (139 Nm)
Writes ~/agile/logs/dr_eval_v2/summary.csv

Usage:
    ~/agile/run_agile.sh scripts/summarize_eval.py [--show_phases]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="~/agile/logs/dr_eval_v2")
parser.add_argument("--policy_dt", type=float, default=0.02)
parser.add_argument("--show_phases", action="store_true")
args = parser.parse_args()

PHASES = {"squat": (3.5, 6.0), "stand_up": (6.5, 9.0), "walk_1.0": (9.5, 19.0), "walk_1.5": (19.5, 24.0), "turn": (24.5, 29.0)}
HEIGHT_PHASES = ["squat", "stand_up"]
WALK_PHASES = ["walk_1.0", "walk_1.5", "turn"]
POLICIES = ["agile", "a", "b", "nominal"]
WORLDS = ["NoTS", "Sat2p0", "Sat1p5", "Sat1p0", "AGILE"]
PAYLOADS = [0, 10, 20]
LOWER_KEYS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
KNEE_TAU_LIM = 139.0

root = Path(args.root).expanduser()


def load(d):
    files = sorted((d / "trajectories").glob("episode_*.parquet"))
    if not files:
        return None, None
    df = pd.concat([pd.read_parquet(f).assign(episode=i) for i, f in enumerate(files)], ignore_index=True)
    df["t"] = df["frame_idx"] * args.policy_dt
    meta = json.loads((d / "trajectories" / "metadata.json").read_text())
    return df, meta["joint_names"]


def phase_rmse(df, phase, col):
    t0, t1 = PHASES[phase]
    sel = df[(df["t"] >= t0) & (df["t"] < t1)]
    return float(np.sqrt(np.mean(sel[col] ** 2))) if len(sel) else np.nan


rows, shown = [], False
for p in POLICIES:
    for w in WORLDS:
        for kg in PAYLOADS:
            df, names = load(root / p / f"{w}_P{kg}")
            if df is None:
                continue
            df["err_vx"] = df["commands_0"] - df["root_lin_vel_robot_0"]
            df["err_h"] = df["commands_3"] - df["root_pos_2"]
            if args.show_phases and not shown:
                print(f"\nMean command per phase ({p}/{w}_P{kg}) [vx, vy, wz, h]:")
                for name, (t0, t1) in PHASES.items():
                    sel = df[(df["t"] >= t0) & (df["t"] < t1)]
                    print(f"  {name:9s}", np.round(sel[[f"commands_{i}" for i in range(4)]].mean().to_numpy(), 2))
                shown = True
            row = {"policy": p, "world": w, "payload_kg": kg,
                   "success": df.groupby("episode")["is_success"].first().mean()}
            for ph in HEIGHT_PHASES:
                row[f"h_rmse_{ph}"] = phase_rmse(df, ph, "err_h")
            for ph in WALK_PHASES:
                row[f"vx_rmse_{ph}"] = phase_rmse(df, ph, "err_vx")
            lower = [j for j, n in enumerate(names) if any(k in n for k in LOWER_KEYS)]
            knees = [j for j, n in enumerate(names) if "knee" in n]
            if f"applied_torque_{lower[0]}" in df.columns:
                late = df[df["t"] >= 1.0]
                app = late[[f"applied_torque_{j}" for j in lower]].to_numpy()
                com = late[[f"computed_torque_{j}" for j in lower]].to_numpy()
                row["clip_%"] = 100.0 * np.mean(np.abs(com - app) > 1.0)
                kn = np.abs(late[[f"applied_torque_{j}" for j in knees]].to_numpy()) / KNEE_TAU_LIM
                row["knee_util_p95"] = float(np.percentile(kn, 95))
            rows.append(row)

if not rows:
    raise SystemExit(f"No results under {root}. Run run_eval_matrix.sh first.")
table = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
out = root / "summary.csv"
table.to_csv(out, index=False)
print(f"\nSaved {out}")
