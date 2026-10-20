"""PPO settings: identical to AGILE's Velocity-Height History runner, only the log names differ."""

from isaaclab.utils.configclass import configclass

from agile.rl_env.tasks.locomotion_height.g1.agents.rsl_rl_ppo_cfg import G1VelocityHeightHistoryPpoRunnerCfg


@configclass
class DRStudyAGILEPpoRunnerCfg(G1VelocityHeightHistoryPpoRunnerCfg):
    experiment_name = "dr_study_g1_agile"
    run_name = "agile_fixed_curve"


@configclass
class DRStudyNominalPpoRunnerCfg(G1VelocityHeightHistoryPpoRunnerCfg):
    experiment_name = "dr_study_g1_nominal"
    run_name = "nominal"


@configclass
class DRStudyAPpoRunnerCfg(G1VelocityHeightHistoryPpoRunnerCfg):
    experiment_name = "dr_study_g1_a"
    run_name = "a_generic_dr"


@configclass
class DRStudyBPpoRunnerCfg(G1VelocityHeightHistoryPpoRunnerCfg):
    experiment_name = "dr_study_g1_b"
    run_name = "b_structure_aware_dr"
