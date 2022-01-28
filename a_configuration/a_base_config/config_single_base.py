import os
import sys

import torch

from g_utils.types import ModelType, LayerActivationType, LossFunctionType


class ConfigBase:
    def __init__(self):
        self.PROJECT_HOME = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
        )
        if self.PROJECT_HOME not in sys.path:
            sys.path.append(self.PROJECT_HOME)

        self.MODEL_SAVE_DIR = os.path.join(self.PROJECT_HOME, "f_play", "models")
        if not os.path.exists(self.MODEL_SAVE_DIR):
            os.mkdir(self.MODEL_SAVE_DIR)

        self.UNITY_ENV_DIR = os.path.join(self.PROJECT_HOME, "b_environments", "unity")
        if not os.path.exists(self.UNITY_ENV_DIR):
            os.mkdir(self.UNITY_ENV_DIR)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.N_STEP = 1

        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 2
        assert self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS >= self.N_VECTORIZED_ENVS * self.N_ACTORS, \
            "TRAIN_INTERVAL_GLOBAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

        self.MAX_TRAINING_STEPS = None

        self.N_EPISODES_FOR_MEAN_CALCULATION = 32
        self.TEST_INTERVAL_TRAINING_STEPS = 1_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.MODEL_PARAMETER = None

        self.N_TEST_EPISODES = 3

        self.CLIP_GRADIENT_VALUE = 10.0

        self.USE_WANDB = False
        self.WANDB_ENTITY = "link-koreatech"

        self.PLAY_MODEL_FILE_NAME = ""

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.SEED = None

        self.SYSTEM_USER_NAME = None

        self.SYSTEM_COMPUTER_NAME = None

        self.USE_LAYER_NORM = False

        self.LAYER_ACTIVATION_TYPE = LayerActivationType.LEAKY_RELU
        self.LAYER_ACTIVATION = None

        self.LOSS_FUNCTION_TYPE = LossFunctionType.HUBER_LOSS
        self.LOSS_FUNCTION = None
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 30

        self.NO_TEST_GRAPHICS = True
