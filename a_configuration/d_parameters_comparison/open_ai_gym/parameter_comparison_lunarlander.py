from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderDqn, \
    ParameterLunarLanderA2c
from g_utils.types import ModelType


class ParameterComparisonLunarLanderDqnRecurrent(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "LunarLander-v2"

        self.AGENT_PARAMETERS = [
            ParameterLunarLanderDqn(),
            ParameterLunarLanderDqn()
        ]

        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_RECURRENT

        self.AGENT_LABELS = [
            "DQN Small Linear",
            "DQN Small Recurrent",
        ]

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

