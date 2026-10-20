"""Check that a DR-study condition puts different actuator values in each env and that the
torque-speed clip has the intended shape.

Usage:
    ~/agile/run_agile.sh scripts/check_dr_study.py --task DRStudy-G1-B-v0
Saves: ~/agile/logs/dr_check/<task>.png
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DRStudy-G1-B-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--joint", type=str, default="left_knee_joint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import agile.rl_env.tasks  # noqa: E402,F401
import agile.isaaclab_extras.monkey_patches  # noqa: E402,F401


def max_torque_curve(act, j, omegas):
    """Max positive torque the actuator allows at each joint velocity, per env (uses the real clip)."""
    saved = act._joint_vel.clone()
    out = []
    for w in omegas:
        act._joint_vel[:] = float(w)
        big = torch.full_like(act._joint_vel, 1.0e9)
        try:
            tau = act._clip_effort(big)
        except Exception:
            v = torch.clip(act._joint_vel, -act._vel_at_effort_lim, act._vel_at_effort_lim)
            top = act._saturation_effort * (1.0 - v / act.velocity_limit)
            tau = torch.minimum(big, torch.clip(top, max=act.effort_limit))
        out.append(tau[:, j].clone())
    act._joint_vel[:] = saved
    return torch.stack(out, dim=1)  # (num_envs, len(omegas))


def find_actuator(robot, joint):
    for name, act in robot.actuators.items():
        if joint in act.joint_names:
            return name, act, act.joint_names.index(joint)
    raise ValueError(f"joint {joint} not found")


def report(robot, joint, tag):
    name, act, j = find_actuator(robot, joint)
    print(f"\n[{tag}] group={name} joint={joint}")
    print(f"  {'env':>4s}{'tau_lim':>10s}{'tau_sat':>12s}{'ratio':>8s}{'w_lim':>8s}{'w_at_lim':>10s}")
    for e in range(act.effort_limit.shape[0]):
        tl = act.effort_limit[e, j].item()
        ts = act._saturation_effort[e, j].item()
        print(f"  {e:4d}{tl:10.2f}{ts:12.4g}{ts / tl:8.2f}{act.velocity_limit[e, j].item():8.2f}"
              f"{act._vel_at_effort_lim[e, j].item():10.2f}")
    return name, act, j


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]

    env.reset()
    _, act, j = report(robot, args_cli.joint, "after reset 1")
    w_lim = act.velocity_limit[0, j].item()
    omegas = torch.linspace(0.0, 1.1 * w_lim, 89)
    curve1 = max_torque_curve(act, j, omegas)

    env.reset()
    report(robot, args_cli.joint, "after reset 2 (values should change for A/B)")

    fig, ax = plt.subplots(figsize=(6, 4))
    for e in range(curve1.shape[0]):
        ax.plot(omegas.numpy(), curve1[e].cpu().numpy(), label=f"env {e}")
    ax.set_xlabel("joint velocity [rad/s]")
    ax.set_ylabel("max available torque [Nm]")
    ax.set_title(f"{args_cli.task} | {args_cli.joint}")
    ax.grid(alpha=0.3)
    ax.legend()
    out_dir = os.path.expanduser("~/agile/logs/dr_check")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args_cli.task}_{args_cli.joint}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved torque-speed plot: {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
