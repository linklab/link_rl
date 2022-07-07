import gym

from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderDqn, \
    ConfigLunarLanderDoubleDqn, ConfigLunarLanderDuelingDqn, ConfigLunarLanderDoubleDuelingDqn
from link_rl.b_environments import wrapper
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.g_utils.types import ModelType


class ConfigComparisonLunarLanderDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "LunarLander-v2"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
            #"Recurrent",
        ]

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDqn(),
            ConfigLunarLanderDqn(),
            #ConfigLunarLanderDqn()
        ]

        # common

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # Linear + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)

        # Recurrent
        #self.AGENT_PARAMETERS[2].MODEL_TYPE = None


class ConfigComparisonLunarLanderDoubleDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "LunarLander-v2"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            #"GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDoubleDqn(),
            #ConfigLunarLanderDoubleDqn()
        ]

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # GRU
        #self.AGENT_PARAMETERS[1].MODEL_TYPE = None


class ConfigComparisonLunarLanderDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "LunarLander-v2"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            "Linear + Time",
            #"Recurrent",
        ]

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDqn(),
            ConfigLunarLanderDqn(),
            #ConfigLunarLanderDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.LunarLanderWithoutVelocity)

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # Linear + Time
        self.AGENT_PARAMETERS[1].MODEL_TYPE = Q_MODEL.QModel.value
        self.AGENT_PARAMETERS[1].WRAPPERS.append(gym.wrappers.TimeAwareObservation)

        # Recurrent
        #self.AGENT_PARAMETERS[2].MODEL_TYPE = None


class ConfigComparisonLunarLanderDoubleDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "LunarLander-v2"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            #"GRU"
        ]

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDoubleDqn(),
            #ConfigLunarLanderDoubleDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.LunarLanderWithoutVelocity)

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # GRU
        #self.AGENT_PARAMETERS[1].MODEL_TYPE = None


class ConfigComparisonLunarLanderDqnTypes(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "LunarLander-v2"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDqn(),
            ConfigLunarLanderDoubleDqn(),
            ConfigLunarLanderDuelingDqn(),
            ConfigLunarLanderDoubleDuelingDqn(),
            ConfigLunarLanderDoubleDuelingDqn(),
        ]

        self.AGENT_PARAMETERS[4].USE_PER = True

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
            "Double Dueling DQN + PER",
        ]
