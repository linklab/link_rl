from a_configuration.parameter_preamble import *


class ParameterComparison(ParameterCartPoleDqn):
    def __init__(self):
        super(ParameterComparison, self).__init__()
        self.USE_WANDB = True
        self.WANDB_ENTITY = "link-koreatech"

        self.parameters = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]

        self.parameters[0].N_STEP = 1
        self.parameters[1].N_STEP = 2
        self.parameters[2].N_STEP = 3

        N_ACTORS = self.parameters[0].N_ACTORS
        assert all(
            parameter.N_ACTORS == N_ACTORS for parameter in self.parameters
        )

        N_VECTORIZED_ENVS = self.parameters[0].N_VECTORIZED_ENVS
        assert all(
            parameter.N_VECTORIZED_ENVS == N_VECTORIZED_ENVS for parameter in self.parameters
        )

        TRAIN_INTERVAL_TOTAL_TIME_STEPS = self.parameters[0].TRAIN_INTERVAL_TOTAL_TIME_STEPS
        assert all(
            parameter.TRAIN_INTERVAL_TOTAL_TIME_STEPS == TRAIN_INTERVAL_TOTAL_TIME_STEPS for parameter in self.parameters
        )

        TEST_INTERVAL_TRAINING_STEPS = self.parameters[0].TEST_INTERVAL_TRAINING_STEPS
        assert all(
            parameter.TEST_INTERVAL_TRAINING_STEPS == TEST_INTERVAL_TRAINING_STEPS for parameter in self.parameters
        )

        CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = self.parameters[0].CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS
        assert all(
            parameter.CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS == CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS for parameter in self.parameters
        )

        N_EPISODES_FOR_MEAN_CALCULATION = self.parameters[0].N_EPISODES_FOR_MEAN_CALCULATION
        assert all(
            parameter.N_EPISODES_FOR_MEAN_CALCULATION == N_EPISODES_FOR_MEAN_CALCULATION for parameter in self.parameters
        )

        MAX_TRAINING_STEPS = self.parameters[0].MAX_TRAINING_STEPS
        assert all(
            parameter.MAX_TRAINING_STEPS == MAX_TRAINING_STEPS for parameter in self.parameters
        )

        N_TEST_EPISODES = self.parameters[0].N_TEST_EPISODES
        assert all(
            parameter.N_TEST_EPISODES == N_TEST_EPISODES for parameter in self.parameters
        )