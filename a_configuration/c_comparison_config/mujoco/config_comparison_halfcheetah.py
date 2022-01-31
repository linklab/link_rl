from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoSac, \
    ConfigHalfCheetahMujocoPpoTrajectory


class ConfigComparisonHalfCheetahMujocoSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "HalfCheetah-v2"

        self.AGENT_PARAMETERS = [
            ConfigHalfCheetahMujocoSac(),
            ConfigHalfCheetahMujocoSac(),
            ConfigHalfCheetahMujocoSac(),
            ConfigHalfCheetahMujocoSac()
        ]

        self.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
        self.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
        self.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
        self.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
        self.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
        self.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
        self.AGENT_LABELS = [
            "alpha = 0.2",
            "alpha = 0.5",
            "alpha tuning (No Alpha Limit)",
            "alpha tuning (Min Alpha = 0.2)",
        ]
        self.MAX_TRAINING_STEPS = 500000
        self.N_RUNS = 5


class ConfigComparisonHalfCheetahMujocoSacPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "HalfCheetah-v2"

        self.AGENT_PARAMETERS = [
            ConfigHalfCheetahMujocoPpoTrajectory(),
            ConfigHalfCheetahMujocoSac()
        ]

        self.AGENT_LABELS = [
            "ppo_trajectory",
            "sac"
        ]
        self.MAX_TRAINING_STEPS = 1_000_000
        self.N_RUNS = 5
