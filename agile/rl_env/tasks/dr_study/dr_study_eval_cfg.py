"""Evaluation "worlds" for the DR study: one fixed lower-body motor model each.

All worlds: no generic DR events, fixed delay, AGILE base tau_lim / omega_lim / gains.
Only the lower-body (legs, feet) torque-speed curve differs:

    NoTS     tau_sat = 1e6                 (the world Nominal was trained in)
    Sat1p0   tau_sat = 1.0 * tau_lim       (harshest, B's lower bound)
    Sat1p5   tau_sat = 1.5 * tau_lim
    Sat2p0   tau_sat = 2.0 * tau_lim       (B's upper bound)
    AGILE    tau_sat = AGILE's own fixed values (legs 180 Nm, feet 80 Nm)

NoTS / AGILE set tau_sat in the config (a single number per group).
Sat* worlds need per-joint values (k * tau_lim, and tau_lim differs per joint), which Isaac Lab's DCMotor
cannot take from the config (it divides by cfg.saturation_effort before AGILE parses it per joint).
So Sat* worlds apply k * tau_lim with the same reset event used by condition B, with range (k, k).
Do not disable events in the eval scenario YAML, or the Sat* worlds fall back to NoTS.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import dr_actuator_events as dr_events
from .dr_study_env_cfg import ALL_ACTUATORS, GENERIC_DR_EVENTS, LOWER_ACTUATORS, G1DRStudyBaseEnvCfg

# Fixed actuator delay in physics steps (0.005 s each). 0 isolates the torque-speed effect.
EVAL_DELAY_STEPS = 0

AGILE_SATURATION = {"legs": 180.0, "feet": 80.0}


def _set_lower_saturation_ratio(cfg, ratio: float) -> None:
    """tau_sat = ratio * tau_lim per joint, applied at every reset (B's event with a fixed ratio)."""
    cfg.events.eval_fixed_saturation_ratio = EventTerm(
        func=dr_events.randomize_saturation_ratio,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "actuator_names": LOWER_ACTUATORS,
            "ratio_range": (ratio, ratio),
        },
    )


@configclass
class G1DRStudyEvalNoTSEnvCfg(G1DRStudyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in ALL_ACTUATORS:
            self.scene.robot.actuators[name].min_delay = EVAL_DELAY_STEPS
            self.scene.robot.actuators[name].max_delay = EVAL_DELAY_STEPS
        for ev in GENERIC_DR_EVENTS:
            if hasattr(self.events, ev):
                setattr(self.events, ev, None)


@configclass
class G1DRStudyEvalSat1p0EnvCfg(G1DRStudyEvalNoTSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_lower_saturation_ratio(self, 1.0)


@configclass
class G1DRStudyEvalSat1p5EnvCfg(G1DRStudyEvalNoTSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_lower_saturation_ratio(self, 1.5)


@configclass
class G1DRStudyEvalSat2p0EnvCfg(G1DRStudyEvalNoTSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_lower_saturation_ratio(self, 2.0)


@configclass
class G1DRStudyEvalAGILEEnvCfg(G1DRStudyEvalNoTSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, value in AGILE_SATURATION.items():
            self.scene.robot.actuators[name].saturation_effort = value
