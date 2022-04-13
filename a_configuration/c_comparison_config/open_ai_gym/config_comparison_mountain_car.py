import gym

from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_mountain_car import ConfigMountainCarDqn, \
    ConfigMountainCarDoubleDqn
from b_environments import wrapper
from g_utils.types import ModelType


class ConfigComparisonMountainCarDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
            "Recurrent",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn()
        ]

        # common

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_LINEAR

        # Linear + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)

        # Recurrent
        self.AGENT_PARAMETERS[2].MODEL_TYPE = ModelType.SMALL_RECURRENT


class ConfigComparisonMountainCarDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
            "Recurrent",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn(),
            ConfigMountainCarDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.MountainCarWithoutVelocity)

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_LINEAR

        # Linear + Time
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)

        # Recurrent
        self.AGENT_PARAMETERS[2].MODEL_TYPE = ModelType.SMALL_RECURRENT


class ConfigComparisonMountainCarDoubleDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "MountainCar-v0"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear without velocity",
            "Recurrent without velocity",
        ]

        self.AGENT_PARAMETERS = [
            ConfigMountainCarDoubleDqn(),
            ConfigMountainCarDoubleDqn(),
            ConfigMountainCarDoubleDqn()
        ]

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_LINEAR

        # Linear without velocity
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[1].WRAPPERS.append(wrapper.MountainCarWithoutVelocity)

        # Recurrent without velocity
        self.AGENT_PARAMETERS[2].MODEL_TYPE = ModelType.SMALL_RECURRENT
        self.AGENT_PARAMETERS[2].WRAPPERS.append(wrapper.MountainCarWithoutVelocity)
