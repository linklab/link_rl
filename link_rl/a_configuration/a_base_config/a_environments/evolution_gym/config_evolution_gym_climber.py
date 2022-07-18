from link_rl.a_configuration.a_base_config.a_environments.evolution_gym import ConfigEvolutionGym


class ConfigEvolutionGymClimberV0(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymClimberV0, self).__init__()
        self.ENV_NAME = "Climber-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000


class ConfigEvolutionGymClimberV1(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymClimberV1, self).__init__()
        self.ENV_NAME = "Climber-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000


class ConfigEvolutionGymClimberV2(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymClimberV2, self).__init__()
        self.ENV_NAME = "Climber-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000
