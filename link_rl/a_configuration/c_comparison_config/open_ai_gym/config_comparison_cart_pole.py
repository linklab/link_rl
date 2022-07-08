import gym

from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn, \
    ConfigCartPoleReinforce, \
    ConfigCartPoleA2c, ConfigCartPoleDoubleDqn, ConfigCartPoleDuelingDqn, ConfigCartPoleDoubleDuelingDqn, \
    ConfigCartPolePpo, ConfigCartPolePpoTrajectory
from link_rl.b_environments import wrapper
from link_rl.c_models_v2.b_q_model import Q_MODEL


class ConfigComparisonCartPoleDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn()
        ]

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 4
        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=4)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDoubleDqn(),
            ConfigCartPoleDuelingDqn(),
            ConfigCartPoleDoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleDqnPer(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDoubleDuelingDqn(),
            ConfigCartPoleDoubleDuelingDqn(),
            ConfigCartPoleDoubleDuelingDqn(),
            ConfigCartPoleDoubleDuelingDqn()
        ]

        self.AGENT_PARAMETERS[1].USE_PER = True
        self.AGENT_PARAMETERS[1].PER_ALPHA = 0.3

        self.AGENT_PARAMETERS[2].USE_PER = True
        self.AGENT_PARAMETERS[2].PER_ALPHA = 0.6

        self.AGENT_PARAMETERS[2].USE_PER = True
        self.AGENT_PARAMETERS[2].PER_ALPHA = 0.9

        self.AGENT_LABELS = [
            "Double Dueling DQN",
            "Double Dueling DQN + PER (ALPHA: 0.3)",
            "Double Dueling DQN + PER (ALPHA: 0.6)",
            "Double Dueling DQN + PER (ALPHA: 0.9)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


# OnPolicy
class ConfigComparisonCartPoleReinforce(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleReinforce(),
            ConfigCartPoleReinforce(),
            ConfigCartPoleReinforce()
        ]


class ConfigComparisonCartPoleA2c(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleA2c(),
            ConfigCartPoleA2c(),
            ConfigCartPoleA2c(),
        ]


class ConfigComparisonCartPolePpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPolePpo(),
            ConfigCartPolePpoTrajectory()
        ]

        self.AGENT_LABELS = [
            "PPO",
            "PPO Trajectory"
        ]
        self.MAX_TRAINING_STEPS = 10_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "CartPole-v1"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Original",
            "Original + Time",
            #"Original + GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            #ConfigCartPoleDqn()
        ]

        # common

        # Original

        # Original + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(
            (gym.wrappers.TimeAwareObservation, {})
        )

        # Original + GRU
        #self.AGENT_PARAMETERS[2].MODEL_TYPE = None


class ConfigComparisonCartPoleDqnRecurrentReversAction(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "CartPole-v1"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Original",
            "Original + Time",
            #"Original + GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            #ConfigCartPoleDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.ReverseActionCartpole)

        # Original

        # Original + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(
            gym.wrappers.TimeAwareObservation
        )

        # Original + GRU
        #self.AGENT_PARAMETERS[2].MODEL_TYPE = None


class ConfigComparisonCartPoleDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "CartPole-v1"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Original",
            "Original + Time",
            #"Original + GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            #ConfigCartPoleDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.CartpoleWithoutVelocity)

        # Original

        # Original + Time
        self.AGENT_PARAMETERS[1].WRAPPERS.append(
            gym.wrappers.TimeAwareObservation
        )

        # Original + GRU
        #self.AGENT_PARAMETERS[2].MODEL_TYPE = None


class ConfigComparisonCartPoleDoubleDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "CartPole-v1"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            #"GRU"
        ]

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDoubleDqn(),
            #ConfigCartPoleDoubleDqn()
        ]

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel

        # GRU
        #self.AGENT_PARAMETERS[1].MODEL_TYPE = None


class ConfigComparisonCartPoleDoubleDqnRecurrentWithoutVelocity(ConfigComparisonBase):
    def __init__(self):
        super().__init__()

        self.ENV_NAME = "CartPole-v1"

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.AGENT_LABELS = [
            "Linear",
            #"GRU"
        ]

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDoubleDqn(),
            #ConfigCartPoleDoubleDqn()
        ]

        # common
        for config in self.AGENT_PARAMETERS:
            config.WRAPPERS.append(wrapper.CartpoleWithoutVelocity)

        # Linear
        self.AGENT_PARAMETERS[0].MODEL_TYPE = Q_MODEL.QModel.value

        # GRU
        #self.AGENT_PARAMETERS[1].MODEL_TYPE = None
