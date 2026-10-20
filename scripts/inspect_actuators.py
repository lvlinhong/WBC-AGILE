"""Print the actuator parameters that are actually used at runtime for an AGILE task.

Usage:
    ~/agile/run_agile.sh scripts/inspect_actuators.py --task Velocity-Height-G1-History-v0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect runtime actuator parameters.")
parser.add_argument("--task", type=str, default="Velocity-Height-G1-History-v0")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import agile.rl_env.tasks  # noqa: E402,F401
import agile.isaaclab_extras.monkey_patches  # noqa: E402,F401


def first_env(x):
    """Return env-0 values as a list, or a short description if not a tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x[0].tolist() if x.dim() == 2 else x.tolist()
    return x


def shape_of(x):
    return tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    sim_dt = env.unwrapped.cfg.sim.dt
    print(f"\n=== Task: {args_cli.task} | sim dt = {sim_dt} s | decimation = {env.unwrapped.cfg.decimation} ===")

    attrs = [
        "effort_limit",
        "velocity_limit",
        "_saturation_effort",
        "_vel_at_effort_lim",
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
    ]

    for group_name, act in robot.actuators.items():
        print(f"\n--- Actuator group: {group_name} ({type(act).__name__}) ---")
        cfg = act.cfg
        print(f"  delay (physics steps): min={getattr(cfg, 'min_delay', None)}, max={getattr(cfg, 'max_delay', None)}"
              f"  -> {getattr(cfg, 'min_delay', 0) * sim_dt * 1e3:.1f}~{getattr(cfg, 'max_delay', 0) * sim_dt * 1e3:.1f} ms")
        for a in attrs:
            print(f"  shape {a:20s}: {shape_of(getattr(act, a, None))}")

        header = f"  {'joint':28s}" + "".join(f"{a.strip('_')[:14]:>15s}" for a in attrs)
        print(header)
        values = {a: first_env(getattr(act, a, None)) for a in attrs}
        for j, name in enumerate(act.joint_names):
            row = f"  {name:28s}"
            for a in attrs:
                v = values[a]
                if isinstance(v, list) and j < len(v):
                    row += f"{v[j]:15.4g}"
                elif isinstance(v, (int, float)):
                    row += f"{v:15.4g}"
                else:
                    row += f"{'-':>15s}"
            print(row)

    # Physics-engine limits, if exposed by this Isaac Lab version.
    for name in ["joint_effort_limits", "joint_vel_limits", "joint_velocity_limits"]:
        v = getattr(robot.data, name, None)
        if v is not None:
            print(f"\nrobot.data.{name} (env 0): shape {shape_of(v)}")
            for jn, val in zip(robot.joint_names, first_env(v)):
                print(f"  {jn:28s}{val:12.4g}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
