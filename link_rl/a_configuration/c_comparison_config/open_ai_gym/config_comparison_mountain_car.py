import gym

from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_mountain_car import ConfigMountainCarDqn, \
    ConfigMountainCarDoubleDqn
from link_rl.b_environments import wrapper
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.g_utils.types import ModelType


class ConfigComparisonMountainCarDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn(),
        ]

        # common

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel

        # Linear + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)



class ConfigComparisonMountainCarDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn(),
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.MountainCarWithoutVelocity)

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # Linear + Time
        self.AGENT_PARAMETERS[1].MODEL_TYPE = Q_MODEL.QModel.value
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)


class ConfigComparisonMountainCarDoubleDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear without velocity",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDoubleDqn(),
            ConfigMountainCarDoubleDqn(),
        ]

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # Linear without velocity
        self.AGENT_PARAMETERS[1].MODEL_TYPE = Q_MODEL.QModel.value
        self.AGENT_PARAMETERS[1].WRAPPERS.append(wrapper.MountainCarWithoutVelocity)
