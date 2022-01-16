from g_utils.types import ModelType


class ParameterRecurrentLinearModel:
    def __init__(self, model_type):
        if model_type == ModelType.SMALL_RECURRENT:
            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 1
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128, 128]
        elif model_type == ModelType.MEDIUM_RECURRENT:
            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 3
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 128]
        elif model_type == ModelType.LARGE_RECURRENT:
            self.HIDDEN_SIZE = 256
            self.NUM_LAYERS = 5
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 256, 128]
        else:
            raise ValueError()
