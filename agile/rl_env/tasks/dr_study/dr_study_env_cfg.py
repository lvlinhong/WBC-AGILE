"""Three training conditions built on AGILE's Velocity-Height History task (robot cfg: unitree_g1.G1_29DOF).

Same in all conditions: robot model, tau_lim, omega_lim, PD gains, armature, action scale,
observations, rewards, network, PPO. Upper-body actuators (waist, arms) keep AGILE values.

            lower-body tau_sat        lower-body tau_lim       generic DR events   delay (physics steps)
Nominal     1e6 (fixed)               base (fixed)             off                 0
A           1e6 (fixed)               s * base, s ~ U(A, per group)       on (AGILE)          0-4 (AGILE)
B           k * base, k ~ U(B)        base (fixed)             on (AGILE)          0-4 (AGILE)
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from agile.rl_env.tasks.locomotion_height.g1.velocity_height_env_cfg import G1VelocityHeightHistoryEnvCfg

from . import dr_actuator_events as dr_events

# Actuator groups whose torque-speed model is the subject of the study (stage 1: legs).
LOWER_ACTUATORS = ["legs", "feet"]
ALL_ACTUATORS = ["legs", "feet", "waist", "arms"]

# "No torque-speed curve". Do not use inf: inf * 0 gives NaN inside the DC-motor clip.
NO_TORQUE_SPEED = 1.0e6

# B: tau_sat / tau_lim range. AGILE's own fixed values fall at 1.29 (hip roll, knee),
# 2.05 (hip pitch, hip yaw) and 1.6 (ankles).
B_RATIO_RANGE = (1.0, 2.0)

# A: tau_lim scale range per actuator group, s_min = 1 - |omega|_95 / omega_lim (omega_lim = 10 rad/s).
# |omega|_95 measured from the Nominal 30k policy (2026-10-17, 32 episodes, first 1 s dropped),
# taking the most demanding joint in each group: legs = hip pitch 3.08 rad/s, feet = ankle roll 3.39 rad/s.
# At that speed, B's worst case (tau_sat = tau_lim) leaves s_min * tau_lim, so A and B cover the same torque range.
A_SCALE_RANGES = {
    "legs": (0.69, 1.0),
    "feet": (0.66, 1.0),
}

# AGILE domain-randomization events that Nominal turns off.
GENERIC_DR_EVENTS = [
    "randomize_physics_material",
    "randomize_lower_actuator_gains",
    "randomize_upper_actuator_gains",
    "randomize_joint_friction",
    "randomize_joint_armature",
    "randomize_bodies_mass",
    "randomize_base_mass",
    "randomize_bodies_com",
    "randomize_base_com",
    "apply_external_force_torque",
    "apply_external_force_torque_extremities",
    "push_robot",
]


@configclass
class G1DRStudyBaseEnvCfg(G1VelocityHeightHistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in LOWER_ACTUATORS:
            self.scene.robot.actuators[name].saturation_effort = NO_TORQUE_SPEED


@configclass
class G1DRStudyNominalEnvCfg(G1DRStudyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in ALL_ACTUATORS:
            self.scene.robot.actuators[name].min_delay = 0
            self.scene.robot.actuators[name].max_delay = 0
        for ev in GENERIC_DR_EVENTS:
            if hasattr(self.events, ev):
                setattr(self.events, ev, None)


@configclass
class G1DRStudyAEnvCfg(G1DRStudyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for group, scale_range in A_SCALE_RANGES.items():
            setattr(
                self.events,
                f"dr_effort_limit_scale_{group}",
                EventTerm(
                    func=dr_events.randomize_effort_limit_scale,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg("robot"),
                        "actuator_names": [group],
                        "scale_range": scale_range,
                    },
                ),
            )


@configclass
class G1DRStudyBEnvCfg(G1DRStudyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.dr_saturation_ratio = EventTerm(
            func=dr_events.randomize_saturation_ratio,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "actuator_names": LOWER_ACTUATORS,
                "ratio_range": B_RATIO_RANGE,
            },
        )
