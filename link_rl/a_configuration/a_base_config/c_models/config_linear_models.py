from link_rl.g_utils.types import ModelType


class ConfigLinearModel:
    def __init__(self, model_type):
        if model_type == ModelType.TINY_LINEAR:
            self.NEURONS_PER_REPRESENTATION_LAYER = [64]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [64]

        elif model_type == ModelType.SMALL_LINEAR:
            self.NEURONS_PER_REPRESENTATION_LAYER = [128]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128, 128]
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128, 128, 128]

        elif model_type == ModelType.SMALL_LINEAR_2:
            self.NEURONS_PER_REPRESENTATION_LAYER = [256]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 256]

        elif model_type == ModelType.MEDIUM_LINEAR:
            self.NEURONS_PER_REPRESENTATION_LAYER = [256]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 128]
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 256, 128]

        elif model_type == ModelType.LARGE_LINEAR:
            self.NEURONS_PER_REPRESENTATION_LAYER = [512]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 256, 128]
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = [512, 256, 256, 128]

        else:
            raise ValueError()