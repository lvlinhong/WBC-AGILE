"""Extra observation terms for evaluation: joint torques (for torque-speed operating-point analysis)."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _to_torch(x) -> torch.Tensor:
    # Isaac Lab 3.0 data fields can be ProxyArray (warp-backed); they expose .torch
    return x.torch if hasattr(x, "torch") else x


def applied_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Torque actually sent to the joints, after the actuator model clipped it."""
    return _to_torch(env.scene[asset_cfg.name].data.applied_torque).clone()


def computed_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Torque the PD controller asked for, before clipping. computed - applied > 0 means the motor limit was active."""
    return _to_torch(env.scene[asset_cfg.name].data.computed_torque).clone()
