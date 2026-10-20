"""Joint-velocity statistics of a policy rollout -> s_min for condition A.

Reads the trajectories saved by eval.py (--run_evaluation --save_trajectories) and prints,
for each lower-body joint, |omega| percentiles and the matched A range:

    s_min = 1 - |omega|_95 / omega_lim

(B's worst case, tau_sat = tau_lim, gives tau_lim * (1 - omega/omega_lim) at speed omega.
 Setting A's lowest scale to that value at |omega|_95 makes both conditions cover the same torque range.)

Usage:
    ~/agile/run_agile.sh scripts/compute_omega95.py ~/agile/logs/evaluation/<folder>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_GROUPS = {
    "legs": ["hip_pitch", "hip_roll", "hip_yaw", "knee"],
    "feet": ["ankle_pitch", "ankle_roll"],
}

parser = argparse.ArgumentParser()
parser.add_argument("eval_dir", type=str, help="logs/evaluation/<task_datetime> folder")
parser.add_argument("--omega_lim", type=float, default=10.0, help="actuator velocity_limit of lower-body joints [rad/s]")
parser.add_argument("--percentile", type=float, default=95.0)
parser.add_argument("--all_episodes", action="store_true", help="include failed episodes too")
parser.add_argument("--skip_s", type=float, default=1.0, help="drop the first seconds of each episode (reset transient)")
parser.add_argument("--policy_dt", type=float, default=0.02, help="seconds per recorded frame (50 Hz policy)")
args = parser.parse_args()

traj_dir = Path(args.eval_dir).expanduser() / "trajectories"
meta = json.loads((traj_dir / "metadata.json").read_text())
joint_names = meta["joint_names"]

files = sorted(traj_dir.glob("episode_*.parquet"))
if not files:
    raise SystemExit(f"No episode_*.parquet in {traj_dir}. Run eval.py with --save_trajectories.")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

n_ep_total = len(files)
if "frame_idx" in df.columns and args.skip_s > 0:
    df = df[df["frame_idx"] >= int(round(args.skip_s / args.policy_dt))]
if not args.all_episodes and "is_success" in df.columns and df["is_success"].any():
    df = df[df["is_success"]]
print(f"Episodes: {n_ep_total} files, frames used: {len(df)} "
      f"({'all episodes' if args.all_episodes else 'successful episodes only'}, first {args.skip_s:.1f} s dropped)")

p = args.percentile
print(f"\n{'joint':28s}{'|w| p50':>10s}{f'|w| p{p:.0f}':>10s}{'|w| p99':>10s}{'|w| max':>10s}{'s_min':>8s}")
group_w = {g: [] for g in LOWER_GROUPS}
for j, name in enumerate(joint_names):
    group = next((g for g, keys in LOWER_GROUPS.items() if any(k in name for k in keys)), None)
    if group is None:
        continue
    col = f"joint_vel_{j}"
    if col not in df.columns:
        continue
    w = np.abs(df[col].to_numpy())
    wp = np.percentile(w, p)
    group_w[group].append(wp)
    s_min = 1.0 - wp / args.omega_lim
    print(f"{name:28s}{np.percentile(w, 50):10.2f}{wp:10.2f}{np.percentile(w, 99):10.2f}{w.max():10.2f}{s_min:8.2f}")

print("\nSuggested A ranges (most demanding joint in each group):")
for g, vals in group_w.items():
    if vals:
        wmax = max(vals)
        print(f"  {g:5s}: max |w|_p{p:.0f} = {wmax:.2f} rad/s  ->  s ~ U({1.0 - wmax / args.omega_lim:.2f}, 1.0)")
