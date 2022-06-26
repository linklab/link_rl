from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoSac, ConfigAntMujocoPpoTrajectory


class ConfigComparisonAntMujocoSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Ant-v2"

        self.AGENT_PARAMETERS = [
            ConfigAntMujocoSac(),
            ConfigAntMujocoSac(),
            ConfigAntMujocoSac(),
            ConfigAntMujocoSac()
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


class ConfigComparisonAntMujocoSacPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Ant-v2"

        self.AGENT_PARAMETERS = [
            ConfigAntMujocoPpoTrajectory(),
            ConfigAntMujocoSac(),
        ]
        self.AGENT_LABELS = [
            "ppo_trajectory",
            "sac",
        ]
        self.MAX_TRAINING_STEPS = 1_000_000
        self.N_RUNS = 5