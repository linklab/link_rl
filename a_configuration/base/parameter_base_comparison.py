import os
import sys

class ParameterComparisonBase:
    def __init__(self):
        self.PROJECT_HOME = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
        )
        if self.PROJECT_HOME not in sys.path:
            sys.path.append(self.PROJECT_HOME)

        self.AGENT_PARAMETERS = None
        self.AGENT_LABELS = None
        
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000

        self.USE_WANDB = False
        self.WANDB_ENTITY = None

        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 4
        assert self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS >= self.N_VECTORIZED_ENVS * self.N_ACTORS, \
            "TRAIN_INTERVAL_GLOBAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

        self.TEST_INTERVAL_TRAINING_STEPS = 256
        self.CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS = 200
        self.N_EPISODES_FOR_MEAN_CALCULATION = 100
        self.MAX_TRAINING_STEPS = 100_000
        self.N_TEST_EPISODES = 3

        self.N_RUNS = 5


        # N_ACTORS = self.AGENT_PARAMETERS[0].N_ACTORS
        # assert all(
        #     parameter.N_ACTORS == N_ACTORS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # N_VECTORIZED_ENVS = self.AGENT_PARAMETERS[0].N_VECTORIZED_ENVS
        # assert all(
        #     parameter.N_VECTORIZED_ENVS == N_VECTORIZED_ENVS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # TRAIN_INTERVAL_GLOBAL_TIME_STEPS = self.AGENT_PARAMETERS[0].TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        # assert all(
        #     parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS == TRAIN_INTERVAL_GLOBAL_TIME_STEPS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # TEST_INTERVAL_TRAINING_STEPS = self.AGENT_PARAMETERS[0].TEST_INTERVAL_TRAINING_STEPS
        # assert all(
        #     parameter.TEST_INTERVAL_TRAINING_STEPS == TEST_INTERVAL_TRAINING_STEPS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS = self.AGENT_PARAMETERS[0].CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS
        # assert all(
        #     parameter.CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS == CONSOLE_LOG_INTERVAL_GLOBAL_TIME_STEPS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # N_EPISODES_FOR_MEAN_CALCULATION = self.AGENT_PARAMETERS[0].N_EPISODES_FOR_MEAN_CALCULATION
        # assert all(
        #     parameter.N_EPISODES_FOR_MEAN_CALCULATION == N_EPISODES_FOR_MEAN_CALCULATION for parameter in self.AGENT_PARAMETERS
        # )
        #
        # MAX_TRAINING_STEPS = self.AGENT_PARAMETERS[0].MAX_TRAINING_STEPS
        # assert all(
        #     parameter.MAX_TRAINING_STEPS == MAX_TRAINING_STEPS for parameter in self.AGENT_PARAMETERS
        # )
        #
        # N_TEST_EPISODES = self.AGENT_PARAMETERS[0].N_TEST_EPISODES
        # assert all(
        #     parameter.N_TEST_EPISODES == N_TEST_EPISODES for parameter in self.AGENT_PARAMETERS
        # )