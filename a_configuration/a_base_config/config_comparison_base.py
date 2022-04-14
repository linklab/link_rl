import os
import sys

import torch
from torch import nn

from g_utils.types import LayerActivationType


class ConfigComparisonBase:
    def __init__(self):
        self.PROJECT_HOME = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
        )
        if self.PROJECT_HOME not in sys.path:
            sys.path.append(self.PROJECT_HOME)

        self.COMPARISON_RESULTS_SAVE_DIR = os.path.join(self.PROJECT_HOME, "e_main", "comparison_results")
        if not os.path.exists(self.COMPARISON_RESULTS_SAVE_DIR):
            os.mkdir(self.COMPARISON_RESULTS_SAVE_DIR)

        self.UNITY_ENV_DIR = os.path.join(self.PROJECT_HOME, "b_environments", "unity")
        if not os.path.exists(self.UNITY_ENV_DIR):
            os.mkdir(self.UNITY_ENV_DIR)

        self.AGENT_PARAMETERS = None
        self.AGENT_LABELS = None
        
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = None

        self.USE_WANDB = False
        self.WANDB_REPORT_URL = None

        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 4
        assert self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS >= self.N_VECTORIZED_ENVS * self.N_ACTORS, \
            "TRAIN_INTERVAL_GLOBAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

        # [NOTE]
        self.TEST_INTERVAL_TRAINING_STEPS = 1_000
        self.N_EPISODES_FOR_MEAN_CALCULATION = 32
        self.MAX_TRAINING_STEPS = 100_000
        self.N_TEST_EPISODES = 3

        self.N_RUNS = 5

        self.USE_LAYER_NORM = False

        self.LAYER_ACTIVATION_TYPE = LayerActivationType.LEAKY_RELU

        self.FORCE_USE_CPU = False
        self.DEVICE = None

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 30
