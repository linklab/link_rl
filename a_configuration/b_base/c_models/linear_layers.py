from g_utils.types import ModelType


class ParameterLinearLayer:
    def __init__(self):
        self.MODEL_TYPE = ModelType.LINEAR


class ParameterSmallLinearLayer(ParameterLinearLayer):
    def __init__(self):
        super(ParameterSmallLinearLayer, self).__init__()
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128, 128]


class ParameterMediumLinearLayer(ParameterLinearLayer):
    def __init__(self):
        super(ParameterMediumLinearLayer, self).__init__()
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 256, 128]


class ParameterLargeLinearLayer(ParameterLinearLayer):
    def __init__(self):
        super(ParameterLargeLinearLayer, self).__init__()
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [512, 256, 256, 128]
