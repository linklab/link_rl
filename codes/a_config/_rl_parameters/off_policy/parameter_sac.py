from codes.a_config._rl_parameters.off_policy.parameter_off_policy import PARAMETERS_OFF_POLICY
from codes.a_config.parameters_general import StochasticActionType, StochasticActionSelectorType


class PARAMETERS_SAC(PARAMETERS_OFF_POLICY):
    ENVIRONMENT_ID = None
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False

    REPLAY_BUFFER_SIZE = 100000

    TAU = 0.001

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = None

    TYPE_OF_STOCHASTIC_ACTION = StochasticActionType.SAMPLE
    TYPE_OF_STOCHASTIC_ACTION_SELECTOR = StochasticActionSelectorType.BASIC_ACTION_SELECTOR

    N_STEP = 2

    ALPHA = 0.2

    ENTROPY_TUNING = False

    LEARNING_RATE = 0.002
    ACTOR_LEARNING_RATE = 0.0002
    ALPHA_LEARNING_RATE = 0.0001

    TRAIN_STEP_FREQ = 2
    POLICY_UPDATE_FREQUENCY = 2 * TRAIN_STEP_FREQ


if __name__ == "__main__":
    params = PARAMETERS_SAC()
    for param in dir(params):
        if not param.startswith("__"):
            print(param, "=", getattr(params, param))