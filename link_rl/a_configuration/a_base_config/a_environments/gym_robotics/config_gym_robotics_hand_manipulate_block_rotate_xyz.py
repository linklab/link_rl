from link_rl.a_configuration.a_base_config.a_environments.gym_robotics import ConfigGymRobotics


class ConfigHandManipulateBlockRotateXYZ(ConfigGymRobotics):
    def __init__(self):
        super(ConfigHandManipulateBlockRotateXYZ, self).__init__()
        self.ENV_NAME = "HandManipulateBlockRotateXYZ-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 990
        self.EPISODE_REWARD_STD_SOLVED = 10.0
