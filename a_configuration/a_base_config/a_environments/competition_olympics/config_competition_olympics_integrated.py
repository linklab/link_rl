from a_configuration.a_base_config.a_environments.competition_olympics import ConfigCompetitionOlympics


class ConfigCompetitionOlympicsIntegrated(ConfigCompetitionOlympics):
    def __init__(self):
        super(ConfigCompetitionOlympics, self).__init__()
        self.ENV_NAME = "olympics-integrated"
        self.CONTROLLED_AGENT_INDEX = 1          # 0 or 1
        self.EPISODE_REWARD_AVG_SOLVED = 990
        self.EPISODE_REWARD_STD_SOLVED = 10.0
