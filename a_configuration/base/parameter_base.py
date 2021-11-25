import os
import sys


class ParameterBase:
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
    )
    if PROJECT_HOME not in sys.path:
        sys.path.append(PROJECT_HOME)

    MODEL_HOME = os.path.join(PROJECT_HOME, "f_play", "models")
    if not os.path.exists(MODEL_HOME):
        os.mkdir(MODEL_HOME)

    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    N_STEP = 1

    TRAIN_INTERVAL_TIME_STEPS = 4
    assert TRAIN_INTERVAL_TIME_STEPS >= N_VECTORIZED_ENVS * N_ACTORS, \
        "TRAIN_INTERVAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

    CONSOLE_LOG_INTERVAL_TIME_STEPS = 100
    TEST_INTERVAL_TIME_STEPS = 1_000

    NUM_EPISODES_FOR_MEAN_CALCULATION = 100

    NUM_TEST_EPISODES = 3

    USE_WANDB = False
    WANDB_ENTITY = "link-koreatech"

