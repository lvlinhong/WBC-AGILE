"""Reset events that randomize the DC-motor torque-speed model per environment.

Both conditions start from the task's base actuator values (AGILE G1_29DOF):
  A (generic DR):          tau_sat = 1e6 (no torque-speed curve), tau_lim <- s * tau_lim_base
  B (structure-aware DR):  tau_lim = tau_lim_base (fixed),       tau_sat <- k * tau_lim_base

After every change, the velocity at which the curve meets tau_lim is recomputed:
  vel_at_effort_lim = velocity_limit * (1 + effort_limit / saturation_effort)
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids, device) -> torch.Tensor:
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        return torch.arange(env.scene.num_envs, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _store_base_values(act) -> None:
    """Keep a private copy of the base values the first time an actuator is touched."""
    if not hasattr(act, "_dr_base_effort_limit"):
        # clone to break any aliasing with effort_limit_sim
        act.effort_limit = act.effort_limit.clone()
        act._saturation_effort = act._saturation_effort.clone()
        act._vel_at_effort_lim = act._vel_at_effort_lim.clone()
        act._dr_base_effort_limit = act.effort_limit.clone()
        act._dr_base_saturation = act._saturation_effort.clone()


def _update_vel_at_effort_lim(act, ids: torch.Tensor) -> None:
    act._vel_at_effort_lim[ids] = act.velocity_limit[ids] * (
        1.0 + act.effort_limit[ids] / act._saturation_effort[ids]
    )


def _sample(n: int, num_joints: int, value_range: tuple[float, float], per_joint: bool, device) -> torch.Tensor:
    shape = (n, num_joints) if per_joint else (n, 1)
    return torch.empty(shape, device=device).uniform_(*value_range)


def randomize_effort_limit_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    actuator_names: list[str],
    scale_range: tuple[float, float],
    per_joint: bool = True,
) -> None:
    """Condition A: scale the torque limit by s ~ U(scale_range). tau_sat is left untouched (1e6)."""
    robot = env.scene[asset_cfg.name]
    for name in actuator_names:
        act = robot.actuators[name]
        _store_base_values(act)
        ids = _resolve_env_ids(env, env_ids, act.effort_limit.device)
        s = _sample(len(ids), act.effort_limit.shape[1], scale_range, per_joint, act.effort_limit.device)
        act.effort_limit[ids] = act._dr_base_effort_limit[ids] * s
        _update_vel_at_effort_lim(act, ids)


def randomize_saturation_ratio(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    actuator_names: list[str],
    ratio_range: tuple[float, float],
    per_joint: bool = True,
) -> None:
    """Condition B: tau_sat = k * tau_lim_base with k ~ U(ratio_range). tau_lim stays at its base value."""
    robot = env.scene[asset_cfg.name]
    for name in actuator_names:
        act = robot.actuators[name]
        _store_base_values(act)
        ids = _resolve_env_ids(env, env_ids, act.effort_limit.device)
        k = _sample(len(ids), act.effort_limit.shape[1], ratio_range, per_joint, act.effort_limit.device)
        act.effort_limit[ids] = act._dr_base_effort_limit[ids]
        act._saturation_effort[ids] = act._dr_base_effort_limit[ids] * k
        _update_vel_at_effort_lim(act, ids)
