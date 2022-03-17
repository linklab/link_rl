from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_frozen_lake import ConfigFrozenLakeDqn

import numpy as np

from g_utils.types import ModelType


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
            "Zero + GRU",
            "Random",
            "Random + Time",
            "Random + GRU",
        ]

        self.AGENT_PARAMETERS = [
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn()
        ]

        # common
        for agent in self.AGENT_PARAMETERS:
            agent.KWARGS["is_slippery"] = False
            agent.KWARGS["desc"] = ["SFF",
                                    "FHF",
                                    "FFG"]

        # Original
        self.AGENT_PARAMETERS[0].TRANSFORM_OBSERVATION = None

        # Zero
        self.AGENT_PARAMETERS[1].TRANSFORM_OBSERVATION = lambda obs: np.zeros(obs.shape)

        # Zero + Time
        self.AGENT_PARAMETERS[2].TRANSFORM_OBSERVATION = lambda obs: np.zeros(obs.shape)
        self.AGENT_PARAMETERS[2].TIME_AWARE_OBSERVATION = True

        # Zero + GRU
        self.AGENT_PARAMETERS[3].TRANSFORM_OBSERVATION = lambda obs: np.zeros(obs.shape)
        self.AGENT_PARAMETERS[3].MODEL_TYPE = ModelType.TINY_RECURRENT

        # Random
        self.AGENT_PARAMETERS[4].TRANSFORM_OBSERVATION = lambda obs: np.random.randn(*obs.shape)

        # Random + Time
        self.AGENT_PARAMETERS[5].TRANSFORM_OBSERVATION = lambda obs: np.random.randn(*obs.shape)
        self.AGENT_PARAMETERS[5].TIME_AWARE_OBSERVATION = True

        # Random + GRU
        self.AGENT_PARAMETERS[6].TRANSFORM_OBSERVATION = lambda obs: np.random.randn(*obs.shape)
        self.AGENT_PARAMETERS[6].MODEL_TYPE = ModelType.TINY_RECURRENT

