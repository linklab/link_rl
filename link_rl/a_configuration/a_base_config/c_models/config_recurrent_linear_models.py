from link_rl.h_utils.types import ModelType


class ConfigRecurrentLinearModel:
    def __init__(self, model_type):
        if model_type == ModelType.TINY_RECURRENT:
            self.NEURONS_PER_REPRESENTATION_LAYER = [64]

            self.HIDDEN_SIZE = 64
            self.NUM_LAYERS = 1

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [64]

        elif model_type == ModelType.SMALL_RECURRENT:
            self.NEURONS_PER_REPRESENTATION_LAYER = [128]

            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 1

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]

        # elif model_type == ModelType.SMALL_RECURRENT:
        #     self.NEURONS_PER_REPRESENTATION_LAYER = [4]
        #
        #     self.HIDDEN_SIZE = 4
        #     self.NUM_LAYERS = 1
        #
        #     self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128, 128, 128]

        elif model_type == ModelType.MEDIUM_RECURRENT:
            self.NEURONS_PER_REPRESENTATION_LAYER = [256]

            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 2

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 128]

        elif model_type == ModelType.LARGE_RECURRENT:
            self.NEURONS_PER_REPRESENTATION_LAYER = [256]

            self.HIDDEN_SIZE = 256
            self.NUM_LAYERS = 3

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256, 128]

        else:
            raise ValueError()
