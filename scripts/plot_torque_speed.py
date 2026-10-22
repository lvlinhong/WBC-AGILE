"""Torque-speed operating points of one joint, one evaluation world, all policies side by side.

Each point is one recorded frame (t >= 1 s). Red points: the motor limit cut the torque (computed - applied > 1 Nm).
Lines: the world's torque-speed envelope  tau_max(w) = min(tau_lim, tau_sat * (1 - w / w_lim)),
                                          tau_min(w) = max(-tau_lim, tau_sat * (-1 - w / w_lim)).

Usage:
    ~/agile/run_agile.sh scripts/plot_torque_speed.py --world Sat1p0_P20 --joint left_knee_joint
Saves: ~/agile/logs/dr_eval_v2/plots/<world>_<joint>.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAU_LIM = {"hip_pitch": 88, "hip_yaw": 88, "hip_roll": 139, "knee": 139, "ankle_pitch": 50, "ankle_roll": 50}
AGILE_SAT = {"hip_pitch": 180, "hip_yaw": 180, "hip_roll": 180, "knee": 180, "ankle_pitch": 80, "ankle_roll": 80}
W_LIM = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="~/agile/logs/dr_eval_v2")
parser.add_argument("--world", type=str, default="Sat1p0_P20", help="<world>_P<kg>, e.g. Sat1p0_P20")
parser.add_argument("--joint", type=str, default="left_knee_joint")
parser.add_argument("--policies", nargs="+", default=["agile", "a", "b"])
parser.add_argument("--policy_dt", type=float, default=0.02)
args = parser.parse_args()

root = Path(args.root).expanduser()
key = next(k for k in TAU_LIM if k in args.joint)
tau_lim = TAU_LIM[key]
world = args.world.split("_P")[0]
tau_sat = {"NoTS": 1e6, "Sat1p0": 1.0 * tau_lim, "Sat1p5": 1.5 * tau_lim, "Sat2p0": 2.0 * tau_lim,
           "AGILE": AGILE_SAT[key]}[world]

w = np.linspace(-W_LIM * 1.1, W_LIM * 1.1, 400)
top = np.minimum(tau_lim, tau_sat * (1 - w / W_LIM))
bot = np.maximum(-tau_lim, tau_sat * (-1 - w / W_LIM))

fig, axes = plt.subplots(1, len(args.policies), figsize=(5 * len(args.policies), 4.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, p in zip(axes, args.policies):
    d = root / p / args.world / "trajectories"
    files = sorted(d.glob("episode_*.parquet"))
    ax.plot(w, top, "k-", lw=1.2)
    ax.plot(w, bot, "k-", lw=1.2)
    ax.set_title(f"{p} | {args.world}")
    ax.set_xlabel(f"{args.joint} velocity [rad/s]")
    ax.grid(alpha=0.3)
    if not files:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    names = json.loads((d / "metadata.json").read_text())["joint_names"]
    j = names.index(args.joint)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["frame_idx"] * args.policy_dt >= 1.0]
    vel, app, com = df[f"joint_vel_{j}"], df[f"applied_torque_{j}"], df[f"computed_torque_{j}"]
    clip = (com - app).abs() > 1.0
    ax.scatter(vel[~clip], app[~clip], s=2, alpha=0.3, c="tab:blue", label="within limit")
    ax.scatter(vel[clip], app[clip], s=3, alpha=0.6, c="tab:red", label=f"clipped ({100 * clip.mean():.1f}%)")
    ax.legend(loc="lower left", fontsize=8)
axes[0].set_ylabel("applied torque [Nm]")
axes[0].set_ylim(-tau_lim * 1.15, tau_lim * 1.15)
fig.tight_layout()
out = root / "plots"
out.mkdir(parents=True, exist_ok=True)
path = out / f"{args.world}_{args.joint}.png"
fig.savefig(path, dpi=120)
print(f"Saved {path}")
