import gym

from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn

import numpy as np

from link_rl.g_utils.types import ModelType


class ConfigComparisonFrozenLakeBase(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "FrozenLake-v1"

        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5

        self.RANDOM_MAP = False
        self.ACTION_MASKING = False
        self.TEST_INTERVAL_TRAINING_STEPS = 500


class ConfigComparisonFrozenLakeDqnActionMasking(ConfigComparisonFrozenLakeBase):
    def __init__(self):
        ConfigComparisonFrozenLakeBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn()
        ]

        self.AGENT_PARAMETERS[0].ACTION_MASKING = True
        self.AGENT_PARAMETERS[0].RANDOM_MAP = False
        self.AGENT_PARAMETERS[1].ACTION_MASKING = False
        self.AGENT_PARAMETERS[1].RANDOM_MAP = False

        self.AGENT_LABELS = [
            "with Action Masking",
            "without Action Masking"
        ]


class ConfigComparisonFrozenLakeDqnTime(ConfigComparisonFrozenLakeBase):
    def __init__(self):
        ConfigComparisonFrozenLakeBase.__init__(self)

        self.AGENT_LABELS = [
            "Original",
            "Zero",
            "Zero + Time",
            #"Zero + GRU",
            "Random",
            "Random + Time",
            #"Random + GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            #ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            #ConfigFrozenLakeDqn()
        ]

        # common
        for agent in self.AGENT_PARAMETERS:
            agent.ENV_KWARGS["is_slippery"] = False
            agent.ENV_KWARGS["desc"] = ["SFF",
                                        "FHF",
                                        "FFG"]

        # Original

        # Zero
        self.AGENT_PARAMETERS[1].WRAPPERS.append(
            (gym.wrappers.TransformObservation, {"f": lambda obs: np.zeros(obs.shape)})
        )

        # Zero + Time
        self.AGENT_PARAMETERS[2].WRAPPERS.append(
            (gym.wrappers.TransformObservation, {"f": lambda obs: np.zeros(obs.shape)})
        )
        self.AGENT_PARAMETERS[2].WRAPPERS.append(
            (gym.wrappers.TimeAwareObservation, {})
        )

        # Zero + GRU
        # self.AGENT_PARAMETERS[3].WRAPPERS.append(
        #     (gym.wrappers.TransformObservation, {"f": lambda obs: np.zeros(obs.shape)})
        # )
        # self.AGENT_PARAMETERS[3].MODEL_TYPE = None

        # Random
        self.AGENT_PARAMETERS[4].WRAPPERS.append(
            (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
        )

        # Random + Time
        self.AGENT_PARAMETERS[5].WRAPPERS.append(
            (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
        )
        self.AGENT_PARAMETERS[5].WRAPPERS.append(
            (gym.wrappers.TimeAwareObservation, {})
        )

        # Random + GRU
        # self.AGENT_PARAMETERS[6].WRAPPERS.append(
        #     (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
        # )
        # self.AGENT_PARAMETERS[6].MODEL_TYPE = None

