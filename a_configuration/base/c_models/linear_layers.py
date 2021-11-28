from g_utils.types import ModelType


class ParameterLinearLayer:
    MODEL_TYPE = ModelType.LINEAR


class ParameterSmallLinearLayer(ParameterLinearLayer):
    NEURONS_PER_LAYER = [128, 128]


class ParameterMediumLinearLayer(ParameterLinearLayer):
    NEURONS_PER_LAYER = [256, 256, 128]


class ParameterLargeLinearLayer(ParameterLinearLayer):
    NEURONS_PER_LAYER = [512, 256, 256, 128]
