import os
import sys


class ParameterBase:
    def __init__(self):
        self.PROJECT_HOME = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
        )
        if self.PROJECT_HOME not in sys.path:
            sys.path.append(self.PROJECT_HOME)

        self.MODEL_SAVE_DIR = os.path.join(self.PROJECT_HOME, "f_play", "models")
        if not os.path.exists(self.MODEL_SAVE_DIR):
            os.mkdir(self.MODEL_SAVE_DIR)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.N_STEP = 1

        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 4
        assert self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS >= self.N_VECTORIZED_ENVS * self.N_ACTORS, \
            "TRAIN_INTERVAL_GLOBAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

        self.CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS = 100

        self.TEST_INTERVAL_TRAINING_STEPS = 256

        self.N_EPISODES_FOR_MEAN_CALCULATION = 32

        self.N_TEST_EPISODES = 3

        self.USE_WANDB = False
        self.WANDB_ENTITY = "link-koreatech"

