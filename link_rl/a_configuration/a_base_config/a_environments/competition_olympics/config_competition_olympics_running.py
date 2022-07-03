from link_rl.a_configuration.a_base_config.a_environments.competition_olympics import ConfigCompetitionOlympics


class ConfigCompetitionOlympicsRunning(ConfigCompetitionOlympics):
    def __init__(self):
        super(ConfigCompetitionOlympics, self).__init__()
        self.ENV_NAME = "olympics-running"
        self.CONTROLLED_AGENT_INDEX = 1          # 0 or 1
        self.N_TEST_EPISODES = 5
        self.EPISODE_REWARD_MIN_SOLVED = 2000
        self.OPPONENT_AGENT_RANDOM_ACTION_RATIO = 'linear(0.7, 0.1, 300000)'
        self.RENDER_OVER_TRAIN = False
        self.RENDER_OVER_TEST = False

        self.GAME_MODE = 0  # 0: running, 1: table_hockey, 2: football, 3: wrestling
        self.FRAME_STACK = 3
