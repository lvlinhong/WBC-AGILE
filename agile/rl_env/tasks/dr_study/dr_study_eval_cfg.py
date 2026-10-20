"""Evaluation "worlds" for the DR study: one fixed lower-body motor model each.

All worlds: no generic DR events, fixed delay, AGILE base tau_lim / omega_lim / gains.
Only the lower-body (legs, feet) torque-speed curve differs:

    NoTS     tau_sat = 1e6                 (the world Nominal was trained in)
    Sat1p0   tau_sat = 1.0 * tau_lim       (harshest, B's lower bound)
    Sat1p5   tau_sat = 1.5 * tau_lim
    Sat2p0   tau_sat = 2.0 * tau_lim       (B's upper bound)
    AGILE    tau_sat = AGILE's own fixed values (legs 180 Nm, feet 80 Nm)

tau_sat is set in the config (not by an event), so it holds even if an eval scenario disables events.
"""

from isaaclab.utils.configclass import configclass

from .dr_study_env_cfg import ALL_ACTUATORS, GENERIC_DR_EVENTS, LOWER_ACTUATORS, G1DRStudyBaseEnvCfg

# Fixed actuator delay in physics steps (0.005 s each). 0 isolates the torque-speed effect.
EVAL_DELAY_STEPS = 0

AGILE_SATURATION = {"legs": 180.0, "feet": 80.0}


def _scaled(value, ratio):
    if isinstance(value, dict):
        return {k: ratio * v for k, v in value.items()}
    return ratio * value


def _set_lower_saturation_ratio(cfg, ratio: float) -> None:
    for name in LOWER_ACTUATORS:
        act = cfg.scene.robot.actuators[name]
        act.saturation_effort = _scaled(act.effort_limit_sim, ratio)


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
