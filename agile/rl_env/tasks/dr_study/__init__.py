"""DR study tasks: Nominal / A (generic DR) / B (structure-aware DR) on the AGILE Velocity-Height G1 task,
plus fixed-motor evaluation worlds."""

import gymnasium as gym

# Training conditions
for _cond in ("Nominal", "A", "B"):
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
for _world in ("NoTS", "Sat1p0", "Sat1p5", "Sat2p0", "AGILE"):
    gym.register(
        id=f"DRStudy-G1-Eval-{_world}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dr_study_eval_cfg:G1DRStudyEval{_world}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:DRStudyNominalPpoRunnerCfg",
        },
    )
