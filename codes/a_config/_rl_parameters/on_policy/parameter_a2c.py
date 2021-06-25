from codes.a_config._rl_parameters.on_policy.parameter_on_policy import PARAMETERS_ON_POLICY


class PARAMETERS_A2C(PARAMETERS_ON_POLICY):
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.001

    ACTION_STD_INIT = 1.0
    ACTION_STD_MIN = 0.01
    ACTION_STD_MIN_STEP = 100000
