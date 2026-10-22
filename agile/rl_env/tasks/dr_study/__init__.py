"""DR study tasks: Nominal / A (generic DR) / B (structure-aware DR) on the AGILE Velocity-Height G1 task,
plus fixed-motor evaluation worlds."""

import gymnasium as gym

# Training conditions
for _cond in ("AGILE", "Nominal", "A", "B"):
    gym.register(
        id=f"DRStudy-G1-{_cond}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dr_study_env_cfg:G1DRStudy{_cond}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:DRStudy{_cond}PpoRunnerCfg",
        },
    )

# Evaluation worlds (used with eval.py --checkpoint <policy>)
#   DRStudy-G1-Eval-<world>-v0        no payload
#   DRStudy-G1-Eval-<world>-P10-v0    +10 kg on the torso
#   DRStudy-G1-Eval-<world>-P20-v0    +20 kg on the torso
for _world in ("NoTS", "Sat1p0", "Sat1p5", "Sat2p0", "AGILE"):
    for _p in ("", "P10", "P20"):
        gym.register(
            id=f"DRStudy-G1-Eval-{_world}{'-' + _p if _p else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.dr_study_eval_cfg:G1DRStudyEval{_world}{_p}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:DRStudyNominalPpoRunnerCfg",
            },
        )
